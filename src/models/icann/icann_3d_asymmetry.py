import tensorflow as tf
from tensorflow import keras
from keras.constraints import Constraint

class ClipConstraint(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        w_no_nan = tf.where(tf.math.is_nan(w), tf.ones_like(w) * tf.sqrt(self.min_value * self.max_value), w)
        return tf.clip_by_value(w_no_nan, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

class Invariants(keras.layers.Layer):
    """
    Layer that computes the three invariants of a diagonal tensor.
    """
    def __init__(self, **kwargs):
        super(Invariants, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # No trainable parameters needed
        super(Invariants, self).build(input_shape)
        
    def call(self, A11, A22, A33):
        I1 = A11 + A22 + A33
        I2 = A11 * A22 + A11 * A33 + A22 * A33
        I3 = A11 * A22 * A33
        return I1, I2, I3
    
class ConvexPolynomialLayer(keras.layers.Layer):
    """
    Polynomial layer: f(x) = sum_{k=1}^K [ a_k * (x + b_k)^(2k) ]
    """
    def __init__(self, K, use_biases=True, l1_reg=0.0, l2_reg=0.01, **kwargs):
        """
        Args:
            K: highest degree of the polynomial (divided by 2)
            use_biases: whether to use trainable biases or set them to zero
            l1_reg: L1 regularization factor
            l2_reg: L2 regularization factor
        """
        super().__init__(**kwargs)
        self.K = K
        self.use_biases = use_biases
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
    def build(self, input_shape):
        # Coefficients for each term
        pos_init = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.5)
        self.a = self.add_weight(
            name='coefficients',
            shape=[self.K],
            initializer=pos_init,
            constraint=tf.keras.constraints.NonNeg(),
            # regularizer=keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            trainable=True
        )
        
        # Bias terms for each polynomial term
        if self.use_biases:
            # Use trainable biases with random initialization
            self.b = self.add_weight(
                name='biases',
                shape=[self.K],
                initializer='random_normal',
                regularizer=keras.regularizers.l2(self.l2_reg),
                trainable=True
            )
        else:
            # Create zeros for biases and make them non-trainable
            self.b = self.add_weight(
                name='biases',
                shape=[self.K],
                initializer='zeros',
                trainable=False
            )
        
        self.built = True
    
    def call(self, x):
        result = tf.zeros_like(x)
        
        for k in range(self.K):
            # Compute (x + b_k)^(2(k+1))
            power = 2 * (k + 1)
            factor = self.a[k] / (k+1)
            input = factor * (x + self.b[k])

            term = tf.pow(input, power)
            
            # Multiply by coefficient a_k
            # term = (term / (4.0 **(k+1)))
            
            # Add to result
            result += term
            
        return result
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'K': self.K,
            'use_biases': self.use_biases,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config

class gNet(keras.layers.Layer):
    """
    Network for generating flow potentials with polynomial activation functions.
    """
    def __init__(self, n, l2_reg=0.01, name=None):
        super().__init__(name=name)
        self.n = n
        self.l2_reg = l2_reg
        self.poly = None
        self.weights_g = None
    
    def build(self, input_shape):
        # Initialize polynomial layer
        self.poly = ConvexPolynomialLayer(self.n, l2_reg=self.l2_reg, use_biases=False)
        self.poly.build(input_shape)
        
        # Initialize weights for activations
        pos_init = tf.keras.initializers.RandomUniform(minval=0.0, maxval=10.0)
        self.weights_g = self.add_weight(
            name='weights_g',
            shape=[7], #7,6
            initializer=pos_init,
            constraint=tf.keras.constraints.NonNeg(),
            # regularizer=keras.regularizers.l1(1e-4),
            trainable=True
        )
        # tf.print("weights_g:", self.weights_g)
        
        self.built = True
    
    def call(self, x):
        # Apply correction terms
        origin = tf.zeros_like(x)
        
        with tf.GradientTape() as correction_tape:
            correction_tape.watch(origin)
            f0 = self.poly(origin)
        
        f0_prime = correction_tape.gradient(f0, origin)
        del correction_tape

        f = self.poly(x)
        p = f - f0 - f0_prime * x

        # tf.print("p:", p)
        
        # Apply activation functions to the polynomial
        p_a1 = tf.math.cosh(p) - 1
        p_a2 = tf.math.log(tf.math.cosh(p))
        p_a3 = tf.math.exp(p) - 1
        p_a4 = tf.pow(p, 2)
        p_a5 = tf.pow(p, 3)
        p_a6 = tf.pow(p, 4)

        

        # Check if p is negative and print a warning if it is
        negative_p = tf.reduce_any(p < 0)
        tf.cond(
            negative_p,
            lambda: tf.print("WARNING: Negative p detected: \n \n", p),
            lambda: tf.no_op()
        )
        
        # Apply the weights for the activation functions
        g = self.weights_g[0] * p

        activations = [p_a1, p_a2, p_a3, p_a4, p_a5, p_a6]
        # activations = [p_a1, p_a2, p_a4, p_a5, p_a6]
        for i, activation in enumerate(activations):
            g += self.weights_g[i+1] * activation
            negative_pa = tf.reduce_any(activation < -1e-9)
            tf.cond(
                negative_pa,
                lambda: tf.print("WARNING: Negative activation {i} detected: \n \n", activation),
                lambda: tf.no_op()
            )
        
        negative_g = tf.reduce_any(g < -1e-9)
        tf.cond(
            negative_g,
            lambda: tf.print("WARNING: Negative g detected: \n \n", g),
            lambda: tf.no_op()
        )
        return g
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n': self.n,
            'l2_reg': self.l2_reg
        })
        return config

class HelmholtzIsoNet(keras.layers.Layer):
    """
    Iso-volumetric part of the Helmholtz free energy network.
    """
    def __init__(self, name=None):
        super().__init__(name=name)
        self.cp1 = None
        self.cp2 = None
        self.cp3 = None

    def build(self, input_shape):
        initializer = tf.keras.initializers.TruncatedNormal(mean=3e-1,   stddev=1.5e-1)#, seed = 42)
        initializer_p = tf.keras.initializers.TruncatedNormal(mean=2e-1, stddev=1e-1  )#, seed = 42)
        # Initialize the weights for the powers
        self.cp1 = self.add_weight(
            name='cp1',
            shape=(2,), 
            initializer=initializer_p, 
            trainable=True, 
            constraint=tf.keras.constraints.NonNeg(),
            regularizer=keras.regularizers.l1(1e-4)
        )

        # Initialize weights for exponential activation
        self.cp2 = self.add_weight(
            name='cp2',
            shape=(2,), 
            initializer=initializer, 
            trainable=True, 
            constraint=tf.keras.constraints.NonNeg(),
            regularizer=keras.regularizers.l1(1e-2)
        )

        # Initialize weights for exponential scaling
        self.cp3 = self.add_weight(
            name='cp3',
            shape=(2,), 
            initializer=initializer, 
            trainable=True, 
            constraint=tf.keras.constraints.NonNeg(),
            regularizer=keras.regularizers.l1(1e-2)
        )
        
        self.built = True

    def call(self, x):
        x2 = tf.pow(x, 2)

        x1exp = self.cp3[0] * tf.exp(self.cp2[0] * x)
        x2exp = self.cp3[1] * tf.exp(self.cp2[1] * x2)

        output = x1exp + x2exp + self.cp1[0] * x + self.cp1[1] * x2
        
        return output
    
class HelmholtzVolNet(keras.layers.Layer):
    """
    Volumetric part of the Helmholtz free energy network.
    """
    def __init__(self, n, name=None):
        super().__init__(name=name)
        self.n = n
        self.power_expansion = None
        self.W3_alpha = None
        self.W3_beta = None

    def build(self, input_shape):
        initializer_a = tf.keras.initializers.TruncatedNormal(mean=0,    stddev=3e-1)#, seed=42)
        initializer_b = tf.keras.initializers.TruncatedNormal(mean=1e-6, stddev=5e-7)#, seed=42)
        # Initialize the power expansion
        self.power_expansion = ConvexPolynomialLayer(self.n, use_biases=False)
        self.power_expansion.build(input_shape)
        
        # Weights for W3
        # self.W3_alpha = self.add_weight(
        #     name='W3_alpha',
        #     shape=[], 
        #     initializer=initializer_a, 
        #     trainable=True 
        # )
        
        # self.W3_beta = self.add_weight(
        #     name='W3_beta',
        #     shape=[], 
        #     initializer=initializer_b, 
        #     trainable=True, 
        #     constraint=tf.keras.constraints.NonNeg()
        # )
        
        self.built = True

    def call(self, x):
        # Compute the power expansion
        p = self.power_expansion(x - 1)
        
        # Safe version of x to prevent log(0) issues
        x_safe = tf.maximum(x, 1e-8)
        
        # Compute the special activation
        # W3 = self.W3_beta * (tf.pow(x_safe, -self.W3_alpha) - 1 + self.W3_alpha * tf.math.log(x_safe))
        
        # Compute the final output
        output = p #+ W3
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'n': self.n})
        return config

class HelmholtzNet(keras.layers.Layer):
    """
    Combined isochoric and volumetric Helmholtz free energy network.
    """
    def __init__(self, n, name=None):
        super().__init__(name=name)
        self.n = n
        
        # Initialize to None, will create in build
        self.iso_net1 = None
        self.iso_net2 = None
        self.vol_net = None

    def build(self, input_shape):
        # Make the build method more robust to handle different input formats
        if isinstance(input_shape, list) and len(input_shape) == 3:
            # Input is already in the correct format
            i1_shape, i2_shape, i3_shape = input_shape
        else:
            # Single tensor shape passed - use it for all three inputs
            i1_shape = i2_shape = i3_shape = input_shape
        
        # Create sublayers with unique names
        self.iso_net1 = HelmholtzIsoNet(name=f"{self.name}_iso1")
        self.iso_net2 = HelmholtzIsoNet(name=f"{self.name}_iso2")
        self.vol_net = HelmholtzVolNet(self.n, name=f"{self.name}_vol")
        
        # Build the sublayers
        self.iso_net1.build(i1_shape)  
        self.iso_net2.build(i2_shape)
        self.vol_net.build(i3_shape)
        
        self.built = True

    def call(self, I1, I2, I3):
        # Safe version of I3 to prevent division by zero
        I3_safe = tf.maximum(I3, 1e-8)
        
        # Compute the corrected invariants
        I1_bar = I1 / (I3_safe ** (1/3))
        I2_bar = I2 / (I3_safe ** (2/3))
        
        # Compute the isotropic part
        iso1 = self.iso_net1(I1_bar - 3)
        iso2 = self.iso_net2(I2_bar - 3)

        # Compute the volumetric part
        vol = self.vol_net(I3)

        # Combine the parts
        output = iso1 + iso2 + vol

        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'n': self.n})
        return config

class piCANN(keras.layers.Layer):  # Change from Model to Layer
    """
    Integrated Constitutive Artificial Neural Network RNN cell.
    """
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n  # List of network sizes [n_e, n_i, n_g1, n_g2, n_g3]
        
        # Define state size for RNN interface - must be class attributes
        self.state_size = [1, 1, 1]  
        self.output_size = 3  # Output size (3 stress components)
        
        # Create layers immediately (don't delay to build)
        self.invariants = Invariants(name="invariants")
        self.hNet_e = HelmholtzNet(self.n[0], name="hNet_e")
        self.hNet_i = HelmholtzNet(self.n[1], name="hNet_i")
        self.gNet1 = gNet(self.n[2], name="gNet1")
        self.gNet2 = gNet(self.n[3], name="gNet2")
        self.gNet3 = gNet(self.n[4], name="gNet3")
        self.tolerance = tf.constant(1e-5, dtype=tf.float32)
        self.lambda_dot = tf.Variable(5e-3, trainable=False, dtype=tf.float32)

        # The yield is slow to learn what the initial yield stress is through the gNets, hopefully this parameter helps
        self.uniform_yield_weight = self.add_weight(
            name='uniform_yield_weight',
            shape=[1],
            initializer=tf.keras.initializers.Constant(3.5),#1.55
            trainable=True,
            constraint=ClipConstraint(0.01, 10.0)
        )
        # self.yield_exponent = tf.constant(16.0, dtype=tf.float32)
    
    def build(self, input_shape):
        # input_shape will be (batch_size, 4) 
        super().build(input_shape)
        self.built = True
        self.hh_list = [self.hNet_e, self.hNet_i]#i
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Generate initial states (identity tensors)."""
        if inputs is not None:
            # If inputs are provided, use their batch size
            batch_size = tf.shape(inputs)[0]
        elif batch_size is None:
            # If neither inputs nor batch_size is provided, use 1 as default
            batch_size = tf.constant(1)
        # Otherwise, use the provided batch_size
        
        # Use dtype or default to float32
        dtype = dtype or tf.float32
        
        return [
            tf.ones((batch_size, 1), dtype=dtype),
            tf.ones((batch_size, 1), dtype=dtype),
            tf.ones((batch_size, 1), dtype=dtype),
        ]
    
    def get_stress(self, variant, inputs):
        net = self.hh_list[variant]
        (C11, C22, C33) = inputs
        with tf.GradientTape(persistent=True) as tape_psi:
            tape_psi.watch([C11, C22, C33])
            I1, I2, I3 = self.invariants(C11, C22, C33)
            psi = net(I1, I2, I3)
        dpsi_11 = tape_psi.gradient(psi, C11)
        dpsi_22 = tape_psi.gradient(psi, C22)
        dpsi_33 = tape_psi.gradient(psi, C33)
        del tape_psi

        return dpsi_11, dpsi_22, dpsi_33,

    def flow(self, inputs, lambda_dot):
        # tf.print("Flowing")
        gamma11, gamma22, gamma33, Ci11, Ci22, Ci33, dt = inputs
        dt = tf.ones_like(dt) 
        with tf.GradientTape(persistent=True) as tape_g:
            tape_g.watch([gamma11, gamma22, gamma33])
            g = self.yield_function([gamma11, gamma22, gamma33])

        
        dg11 = tape_g.gradient(g, gamma11)
        dg22 = tape_g.gradient(g, gamma22)
        dg33 = tape_g.gradient(g, gamma33)

        del tape_g

        dgm  = tf.maximum(tf.maximum(tf.abs(dg11), tf.abs(dg22)), tf.abs(dg33)) + 1e-4

        D11 = lambda_dot * dg11 / dgm # Normalise to obtain reasonable initial guess for lambda_dot
        D22 = lambda_dot * dg22 / dgm # Normalise to obtain reasonable initial guess for lambda_dot
        D33 = lambda_dot * dg33 / dgm # Normalise to obtain reasonable initial guess for lambda_dot
        # Update inelastic strains
        Ci11_new = Ci11 * tf.exp(2 * dt * D11)
        Ci22_new = Ci22 * tf.exp(2 * dt * D22)
        Ci33_new = Ci33 * tf.exp(2 * dt * D33)
        return Ci11_new, Ci22_new, Ci33_new
    
    def yield_function(self, inputs):
        gamma11, gamma22, gamma33 = inputs
        I1, I2, I3 = self.invariants(gamma11*self.uniform_yield_weight, 
                                     gamma22*self.uniform_yield_weight, 
                                     gamma33*self.uniform_yield_weight)
        J2 = tf.pow(I1, 2)  - 2 * I2
        J3 = tf.pow(I1, 3) - 3 * I1 * I2 + 3 * I3
        g1 = 0.0* self.gNet1(I1)
        g2 = self.gNet2(J2)
        g3 = 0.0* self.gNet3(J3)
        Psi = g1 + g2 + g3 - 1

        # g1 = tf.pow(tf.abs(gamma11), self.yield_exponent)
        # g2 = tf.pow(tf.abs(gamma22), self.yield_exponent)
        # g3 = tf.pow(tf.abs(gamma33), self.yield_exponent)
        # g  = self.uniform_yield_weight * tf.pow(g1 + g2 + g3, 1/self.yield_exponent)
        # Psi = g - 1.0
        return Psi 
    
    def call(self, inputs, states=None, training=None):
        if states is None:
            states = self.get_initial_state(batch_size=tf.shape(inputs)[0])
            
        # Extract inputs
        batch_size = tf.shape(inputs)[0]
        C11 = inputs[:, 0:1]  
        C22 = inputs[:, 1:2]
        C33 = inputs[:, 2:3]
        dt = inputs[:, 3:4]
        
        # Use states as internal variables
        Ci11, Ci22, Ci33 = states
        
        # Compute elastic Cauchy-Green strain tensor
        Ce11 = C11 / Ci11
        Ce22 = C22 / Ci22
        Ce33 = C33 / Ci33

        # Get derivatives of the Helmholtz free energy
        dpsi_e_11, dpsi_e_22, dpsi_e_33 = self.get_stress(0, [Ce11, Ce22, Ce33])
        dpsi_i_11, dpsi_i_22, dpsi_i_33 = self.get_stress(1, [Ci11, Ci22, Ci33])

        def stress_components(inputs):
            dpsi_e_11, dpsi_e_22, dpsi_e_33, dpsi_i_11, dpsi_i_22, dpsi_i_33 = inputs
            mandel11 = 2 * Ce11 * dpsi_e_11
            mandel22 = 2 * Ce22 * dpsi_e_22
            mandel33 = 2 * Ce33 * dpsi_e_33

            back_stress11 = 2 * dpsi_i_11 * Ci11
            back_stress22 = 2 * dpsi_i_22 * Ci22
            back_stress33 = 2 * dpsi_i_33 * Ci33

            return mandel11, mandel22, mandel33, back_stress11, back_stress22, back_stress33

        # Get Mandel and back stress
        mandel11, mandel22, mandel33, back_stress11, back_stress22, back_stress33 = stress_components([dpsi_e_11, dpsi_e_22, dpsi_e_33, dpsi_i_11, dpsi_i_22, dpsi_i_33])

        # Compute relative stress
        gamma11 = mandel11 - back_stress11
        gamma22 = mandel22 - back_stress22
        gamma33 = mandel33 - back_stress33

        # Compute yield function, associated flow rule h = g
        Psi = self.yield_function([gamma11, gamma22, gamma33])
        yielding = Psi > self.tolerance

        def body(i, lambda_dot, gamma11_curr, gamma22_curr, gamma33_curr):
            with tf.GradientTape(persistent=True) as tape_lambda:
                tape_lambda.watch(lambda_dot)
                # Use the current state values in flow calculation
                Ci11_new, Ci22_new, Ci33_new = self.flow(
                    [gamma11_curr, gamma22_curr, gamma33_curr, Ci11, Ci22, Ci33, dt], 
                    lambda_dot
                )

                # Calculate updated elastic strains
                Ce11_new = C11 / Ci11_new
                Ce22_new = C22 / Ci22_new
                Ce33_new = C33 / Ci33_new

                # Get stresses for updated configuration
                dpsi_e_11, dpsi_e_22, dpsi_e_33 = self.get_stress(0, [Ce11_new, Ce22_new, Ce33_new])
                dpsi_i_11, dpsi_i_22, dpsi_i_33 = self.get_stress(1, [Ci11_new, Ci22_new, Ci33_new])

                # Get stress components
                mandel11, mandel22, mandel33, back_stress11, back_stress22, back_stress33 = stress_components(
                    [dpsi_e_11, dpsi_e_22, dpsi_e_33, dpsi_i_11, dpsi_i_22, dpsi_i_33]
                )

                # Calculate updated gamma values
                gamma11_new = mandel11 - back_stress11
                gamma22_new = mandel22 - back_stress22
                gamma33_new = mandel33 - back_stress33

                # Compute yield function with updated values
                Psi = self.yield_function([gamma11_new, gamma22_new, gamma33_new])
                # tf.print("Psi, lambda:", Psi, lambda_dot)
                minimiser = tf.pow(Psi+0.25 * self.tolerance, 1) # err on the side of caution
            
            # Compute gradient of minimiser with respect to lambda_dot
            dPsidlambda = tape_lambda.gradient(minimiser, lambda_dot)
            del tape_lambda
            
            # Prevent division by zero
            dPsidlambda = tf.where(
                tf.abs(dPsidlambda) < minimiser,
                minimiser*tf.sign(dPsidlambda),
                dPsidlambda
            )

            
            # Update lambda_dot using Newton-Raphson
            update_factor = minimiser / dPsidlambda  
            update_factor = tf.where(tf.abs(update_factor) > tf.abs(lambda_dot), tf.abs(lambda_dot)*tf.sign(update_factor), update_factor)
            damping = tf.constant(0.9, dtype=tf.float32)
            lambda_dot_new = lambda_dot - damping * update_factor
            
            
            return [i + 1, lambda_dot_new, gamma11_new, gamma22_new, gamma33_new]

        def condition(i, lambda_dot, gamma11_curr, gamma22_curr, gamma33_curr):
            # Check if yield function is within tolerance
            Psi = self.yield_function([gamma11_curr, gamma22_curr, gamma33_curr])
            return tf.math.logical_or(tf.reduce_any(tf.abs(Psi) > self.tolerance), tf.reduce_any(Psi > 0))

        # Define the two computation paths
        def newton_raphson_path():
            # let's ignore batch handling for now
            lambda_dot = tf.ones_like(gamma11) * self.lambda_dot

            # Define the two branches for tf.cond
            i = tf.constant(0)
            # Run Newton-Raphson iterations
            [final_it, lambda_dot_final, gamma11_final, gamma22_final, gamma33_final] = tf.while_loop(
                condition, body, [i, lambda_dot, gamma11, gamma22, gamma33],
                maximum_iterations=50
            )
            psi_final = self.yield_function([gamma11_final, gamma22_final, gamma33_final])
            # tf.print("Newton stopped at iteration:", final_it, "with psi:", psi_final, "and lambda_dot:", lambda_dot_final)
            
            lambda_dot      = lambda_dot_final
            gamma11_updated = gamma11_final
            gamma22_updated = gamma22_final
            gamma33_updated = gamma33_final

            self.lambda_dot.assign(tf.reduce_mean(lambda_dot_final))  # Update the class variable
            
                        
            # Compute flow with properly masked lambda_dot (zeros for non-yielding)
            Ci11_new, Ci22_new, Ci33_new = self.flow(
                [gamma11_updated, gamma22_updated, gamma33_updated, Ci11, Ci22, Ci33, dt], 
                lambda_dot
            )

            # Rest of the function stays the same
            Ce11_new = C11 / Ci11_new
            Ce22_new = C22 / Ci22_new
            Ce33_new = C33 / Ci33_new
            
            dpsi_e_11_new, dpsi_e_22_new, dpsi_e_33_new = self.get_stress(0, [Ce11_new, Ce22_new, Ce33_new])
            
            return Ci11_new, Ci22_new, Ci33_new, dpsi_e_11_new, dpsi_e_22_new, dpsi_e_33_new

        def no_yielding_path():
            # Return exactly the same tensor types and shapes as newton_raphson_path
            dpsi_e_11_current, dpsi_e_22_current, dpsi_e_33_current = self.get_stress(0, [Ce11, Ce22, Ce33])
            return Ci11, Ci22, Ci33, dpsi_e_11_current, dpsi_e_22_current, dpsi_e_33_current

        # Use tf.cond for graph-compatible conditional execution
        Ci11_new, Ci22_new, Ci33_new, dpsi_e_11, dpsi_e_22, dpsi_e_33 = tf.cond(
            tf.reduce_any(yielding),
            newton_raphson_path,
            no_yielding_path
        )

        # Compute Second Piola-Kirchhoff stress
        S11 = 2 * dpsi_e_11 / Ci11_new
        S22 = 2 * dpsi_e_22 / Ci22_new
        S33 = 2 * dpsi_e_33 / Ci33_new

        # Convert PK2 to Cauchy stress
        # sigma = J^-1 F S F^T
        # J = det(F) = sqrt(det(C))
        Cauchy11 = S11 * C11 / tf.sqrt((C11 * C22 *C33))
        Cauchy22 = S22 * C22 / tf.sqrt((C11 * C22 *C33))
        Cauchy33 = S33 * C33 / tf.sqrt((C11 * C22 *C33))
        
        # Concatenate the stress outputs into a single tensor (required for RNN interface)
        # output = tf.concat([S11, S22, S33], axis=-1)
        output = tf.concat([Cauchy11, Cauchy22, Cauchy33], axis=-1)
        
        # Return format that RNN expects: [output, new_states]
        return output, [Ci11_new, Ci22_new, Ci33_new]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n': self.n
        })
        return config
    
def create_rnn_model(n):
    # Create and compile the model before use
    icann = piCANN(n) # Change this depending on what model you are interested in
    
    # Use None for variable sequence length
    C11 = keras.layers.Input(shape=(None, 1), name='C11')  # None = any sequence length
    C22 = keras.layers.Input(shape=(None, 1), name='C22')
    C33 = keras.layers.Input(shape=(None, 1), name='C33')
    dt = keras.layers.Input(shape=(None, 1), name='dt')
    
    # Concatenate inputs along feature dimension
    inputs = keras.layers.Concatenate(axis=2)([C11, C22, C33, dt])
    
    # Let the RNN cell handle initial states automatically (no explicit initial_state)
    outputs = keras.layers.RNN(
        icann, 
        return_sequences=True, 
        return_state=True,
        stateful=False,
        name='rnn_call'
    )(inputs)
    
    # Split outputs - first is sequence output, rest are states
    sequence_output, *states = outputs
    
    # Split the sequence output into S11, S22, S33
    S11 = keras.layers.Lambda(lambda x: x[:,:,0:1], name='S11')(sequence_output)
    S22 = keras.layers.Lambda(lambda x: x[:,:,1:2], name='S22')(sequence_output)
    S33 = keras.layers.Lambda(lambda x: x[:,:,2:3], name='S33')(sequence_output)
    
    # Build model
    model = keras.Model(
        inputs=[C11, C22, C33, dt], 
        outputs=[S11, S22, S33] + states
    )
    
    return model

if __name__ == '__main__':
    # Direct example usage
    # Initialize the iCANN model with network sizes
    n = [5, 5, 5, 5, 5]  # List of 5 values for network configurations
    icann_model = piCANN(n)

    batch_size = 2

    # Create example inputs (batch size = 32)
    C11 = tf.ones((batch_size, 1))*0.98  # First diagonal component of Cauchy-Green tensor
    C22 = tf.ones((batch_size, 1))*(1.0/0.98)  # Second diagonal component
    C33 = tf.ones((batch_size, 1))*1.00  # Third diagonal component
    dt = 0.01 * tf.ones((batch_size, 1))  # Time step

    # Initial internal states (initially identity matrix)
    Ci11 = tf.ones((batch_size, 1))
    Ci22 = tf.ones((batch_size, 1))
    Ci33 = tf.ones((batch_size, 1))
    states = [Ci11, Ci22, Ci33]

    inputs = tf.concat([C11, C22, C33, dt], axis=1)
    output, new_states = icann_model(inputs, states)

    # Extract individual stress components from the concatenated output
    S11 = output[:, 0:1]
    S22 = output[:, 1:2]
    S33 = output[:, 2:3]
    Ci11_new, Ci22_new, Ci33_new = new_states

    print(f"Stresses: {S11.shape}, {S22.shape}, {S33.shape}")
    print(f"Updated internal states: {Ci11_new.shape}, {Ci22_new.shape}, {Ci33_new.shape}")
    print("Max(S11):", tf.reduce_max(S11).numpy())
    print("Max(S22):", tf.reduce_max(S22).numpy())
    print("Max(S33):", tf.reduce_max(S33).numpy())

    # RNN example usage
    # Create the RNN model
    n = [4, 4, 4, 4, 4]  # Network sizes
    tf.random.set_seed(4)
    rnn_model = create_rnn_model(n)

    # Create a time sequence of deformation data (batch_size=32, time_steps=10)
    batch_size = 1
    time_steps = 100

    # Example strain history sequence
    C11_seq = tf.ones((batch_size, time_steps, 1)) * tf.reshape(tf.linspace(1.0, 0.45, time_steps), (1, time_steps, 1))**2
    C22_seq = tf.pow(C11_seq, -1)
    C33_seq = tf.ones((batch_size, time_steps, 1))
    dt_seq = 0.1 * tf.ones((batch_size, time_steps, 1))

    # Initial states (identity)
    initial_state = [tf.ones((batch_size, 1)), tf.ones((batch_size, 1)), tf.ones((batch_size, 1))]

    # Feed data into the model
    outputs = rnn_model.predict({
        'C11': C11_seq, 
        'C22': C22_seq, 
        'C33': C33_seq, 
        'dt': dt_seq
    })

    # Extract stress history (first 3 outputs) and final internal state (last 3 outputs)
    stress_history = outputs[:3]
    final_state = outputs[3:]

    print(f"Stress history shape: {[s.shape for s in stress_history]}")
    print(f"Final internal state shape: {[s.shape for s in final_state]}")
    print("Max(S11):", tf.reduce_min(stress_history[0]).numpy())
    print("Max(S22):", tf.reduce_min(stress_history[1]).numpy())
    print("Max(S33):", tf.reduce_min(stress_history[2]).numpy())
    print("Final Ci11:", tf.reduce_min(final_state[0]).numpy())
    print("Final Ci22:", tf.reduce_min(final_state[1]).numpy())
    print("Final Ci33:", tf.reduce_min(final_state[2]).numpy())