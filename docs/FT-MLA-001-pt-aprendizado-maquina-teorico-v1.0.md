# FT-MLA-001: Fine-Tuning para IA em Aprendizado de Máquina Teórico

## Visão Geral do Projeto

Este documento estabelece diretrizes para o fine-tuning de modelos de IA especializados nos fundamentos teóricos do aprendizado de máquina, abrangendo desde princípios matemáticos básicos até algoritmos avançados de otimização e teoria da aprendizagem estatística.

### Contexto Filosófico
O aprendizado de máquina teórico representa a ponte entre a estatística matemática e a computação prática, buscando compreender os limites fundamentais do que é aprendível a partir de dados e como otimizar esse processo de aprendizagem.

### Metodologia de Aprendizado Recomendada
1. **Fundamentos Primeiro**: Dominar teoria da probabilidade e otimização convexa
2. **Abordagem Matemática**: Enfatizar rigor matemático e derivações formais
3. **Conexões Interdisciplinares**: Relacionar com estatística, otimização e teoria da informação
4. **Implementação Prática**: Conectar teoria com algoritmos implementáveis
5. **Análise de Limites**: Compreender trade-offs e limitações teóricas

---

## 1. FUNDAMENTOS MATEMÁTICOS E PROBABILÍSTICOS

### 1.1 Espaços de Hilbert e Geometria Convexa
```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class HilbertSpaceAnalysis:
    """
    Análise em espaços de Hilbert para aprendizado de máquina
    """

    def __init__(self, dimension=2):
        self.dimension = dimension
        self.basis_vectors = self._create_orthonormal_basis()

    def _create_orthonormal_basis(self):
        """Cria base ortonormal canônica"""
        basis = []
        for i in range(self.dimension):
            e_i = np.zeros(self.dimension)
            e_i[i] = 1.0
            basis.append(e_i)
        return np.array(basis)

    def gram_schmidt_orthogonalization(self, vectors):
        """
        Processo de Gram-Schmidt para ortogonalização
        """
        orthogonal_vectors = []
        for v in vectors:
            u = np.array(v, dtype=float)

            # Subtrair projeções nos vetores anteriores
            for u_prev in orthogonal_vectors:
                projection = (np.dot(u, u_prev) / np.dot(u_prev, u_prev)) * u_prev
                u = u - projection

            # Normalizar
            if np.linalg.norm(u) > 1e-10:
                u = u / np.linalg.norm(u)
                orthogonal_vectors.append(u)

        return np.array(orthogonal_vectors)

    def reproducing_kernel_hilbert_space(self, kernel_function):
        """
        Análise em espaços de Hilbert com kernel reprodutor (RKHS)
        """
        class RKHS:
            def __init__(self, kernel):
                self.kernel = kernel
                self.training_points = []
                self.alpha_coefficients = []

            def fit(self, X, y):
                """Ajuste usando kernel ridge regression"""
                self.training_points = X

                # Construir matriz de kernel
                K = np.zeros((len(X), len(X)))
                for i in range(len(X)):
                    for j in range(len(X)):
                        K[i, j] = self.kernel(X[i], X[j])

                # Regularização (ridge parameter)
                lambda_reg = 0.01
                K_reg = K + lambda_reg * np.eye(len(X))

                # Resolver sistema
                self.alpha_coefficients = np.linalg.solve(K_reg, y)

            def predict(self, x_test):
                """Predição usando representação kernel"""
                predictions = []
                for x in x_test:
                    prediction = 0
                    for i, x_train in enumerate(self.training_points):
                        prediction += self.alpha_coefficients[i] * self.kernel(x, x_train)
                    predictions.append(prediction)

                return np.array(predictions)

            def representer_theorem_verification(self):
                """Verifica teorema do representador"""
                # Toda função no RKHS pode ser escrita como combinação linear
                # dos valores do kernel avaliados nos pontos de treinamento
                return "f(x) = Σ α_i K(x, x_i)"

        return RKHS(kernel_function)

    def functional_analysis_optimization(self, objective_function, constraint_set):
        """
        Otimização em espaços funcionais usando análise funcional
        """
        # Exemplo: Problema de otimização convexa em Hilbert
        def constrained_optimization(f, C):
            """
            Min f(x) sujeito a x ∈ C
            Usando método do gradiente projetado
            """

            # Inicialização
            x_current = np.random.randn(self.dimension)
            learning_rate = 0.01
            max_iterations = 1000

            for iteration in range(max_iterations):
                # Calcular gradiente
                gradient = self._numerical_gradient(f, x_current)

                # Passo do gradiente
                x_new = x_current - learning_rate * gradient

                # Projeção no conjunto viável
                x_new = self._project_onto_constraint_set(x_new, C)

                # Verificar convergência
                if np.linalg.norm(x_new - x_current) < 1e-6:
                    break

                x_current = x_new

            return x_current

        return constrained_optimization(objective_function, constraint_set)

    def _numerical_gradient(self, f, x, h=1e-5):
        """Gradiente numérico"""
        gradient = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h

            gradient[i] = (f(x_plus) - f(x_minus)) / (2 * h)

        return gradient

    def _project_onto_constraint_set(self, x, C):
        """Projeção no conjunto de restrições"""
        # Implementação simplificada para conjunto convexo
        if C == 'unit_ball':
            # Projeção na bola unitária
            norm_x = np.linalg.norm(x)
            if norm_x > 1:
                return x / norm_x
            else:
                return x
        else:
            return x  # Sem projeção

    def banach_fixed_point_theorem_application(self, contraction_mapping, initial_point):
        """
        Aplicação do teorema do ponto fixo de Banach em aprendizado
        """
        def fixed_point_iteration(T, x0, tolerance=1e-6, max_iter=100):
            """
            Iteração do ponto fixo para mapeamentos contractivos
            """
            x_current = x0
            iterations = []

            for i in range(max_iter):
                x_new = T(x_current)
                iterations.append(x_new)

                if np.linalg.norm(x_new - x_current) < tolerance:
                    break

                x_current = x_new

            return x_current, iterations

        fixed_point, iteration_history = fixed_point_iteration(
            contraction_mapping, initial_point
        )

        return {
            'fixed_point': fixed_point,
            'convergence_history': iteration_history,
            'convergence_rate': len(iteration_history)
        }
```

**Conceitos Críticos:**
- Espaços de Hilbert e geometria convexa
- Processo de Gram-Schmidt e bases ortonormais
- Espaços de Hilbert com kernel reprodutor (RKHS)
- Teorema do representador
- Teorema do ponto fixo de Banach

### 1.2 Otimização Convexa e Não-Convexa
```python
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

class ConvexOptimizationTheory:
    """
    Teoria da otimização convexa para aprendizado de máquina
    """

    def __init__(self):
        self.convergence_history = []

    def convex_function_properties(self, function, domain):
        """
        Propriedades de funções convexas
        """
        class ConvexFunction:
            def __init__(self, f, domain):
                self.f = f
                self.domain = domain

            def is_convex(self):
                """Verifica convexidade através de condições suficientes"""
                # Teste simplificado: função quadrática é convexa
                return self._test_convexity()

            def _test_convexity(self):
                """Teste numérico de convexidade"""
                # Amostrar pontos no domínio
                test_points = np.random.uniform(
                    self.domain[0], self.domain[1], 100
                )

                convex_violations = 0

                for i in range(len(test_points)):
                    for j in range(i+1, len(test_points)):
                        for lambda_val in [0.3, 0.7]:  # Testar convexidade
                            x1, x2 = test_points[i], test_points[j]

                            # Desigualdade de Jensen
                            lhs = self.f(lambda_val * x1 + (1-lambda_val) * x2)
                            rhs = lambda_val * self.f(x1) + (1-lambda_val) * self.f(x2)

                            if lhs > rhs + 1e-6:  # Tolerância numérica
                                convex_violations += 1

                return convex_violations == 0

            def subdifferential(self, x):
                """Subdiferencial de funções convexas"""
                # Para funções diferenciáveis, subdiferencial = {f'(x)}
                # Implementação simplificada
                h = 1e-5
                derivative = (self.f(x + h) - self.f(x - h)) / (2 * h)
                return [derivative]  # Retornar como conjunto

            def conjugate_function(self, x):
                """Função conjugada (Fenchel-Legendre)"""
                def conjugate(y):
                    # f*(y) = sup_x (⟨x,y⟩ - f(x))
                    def objective(x_val):
                        return np.dot(x_val, y) - self.f(x_val)

                    # Maximização numérica
                    result = minimize_scalar(
                        lambda t: -objective(np.array([t])),
                        bounds=(-10, 10),
                        method='bounded'
                    )

                    return -result.fun

                return conjugate(x)

        return ConvexFunction(function, domain)

    def lagrangian_duality(self, objective_function, constraints):
        """
        Dualidade Lagrangiana e teoria dos multiplicadores
        """
        class LagrangianAnalysis:
            def __init__(self, f, constraints):
                self.f = f
                self.constraints = constraints

            def lagrangian_function(self, x, lambda_params):
                """Função Lagrangiana L(x,λ) = f(x) + Σ λ_i g_i(x)"""
                L = self.f(x)

                for i, constraint in enumerate(self.constraints):
                    g_i = constraint['function']
                    lambda_i = lambda_params[i]
                    L += lambda_i * g_i(x)

                return L

            def dual_function(self, lambda_params):
                """Função dual g(λ) = inf_x L(x,λ)"""
                def dual_objective(x):
                    return self.lagrangian_function(x, lambda_params)

                # Minimização numérica
                result = minimize_scalar(dual_objective, bounds=(-10, 10), method='bounded')

                return result.fun

            def kkt_conditions(self, x_opt, lambda_opt):
                """Condições KKT (Karush-Kuhn-Tucker)"""
                conditions = {}

                # Estacionariedade
                conditions['stationarity'] = self._check_stationarity(x_opt, lambda_opt)

                # Viabilidade primal
                conditions['primal_feasibility'] = all(
                    constraint['function'](x_opt) <= 0
                    for constraint in self.constraints
                )

                # Viabilidade dual
                conditions['dual_feasibility'] = all(l >= 0 for l in lambda_opt)

                # Complementariedade
                conditions['complementarity_slackness'] = all(
                    abs(lambda_i * constraint['function'](x_opt)) < 1e-6
                    for lambda_i, constraint in zip(lambda_opt, self.constraints)
                )

                return conditions

            def _check_stationarity(self, x, lambda_params):
                """Verifica condição de estacionariedade"""
                # ∇L(x,λ) = 0
                h = 1e-5
                grad_L = (self.lagrangian_function(x + h, lambda_params) -
                         self.lagrangian_function(x - h, lambda_params)) / (2 * h)

                return abs(grad_L) < 1e-6

        return LagrangianAnalysis(objective_function, constraints)

    def interior_point_methods(self, objective_function, constraints):
        """
        Métodos de pontos interiores para otimização convexa
        """
        class InteriorPointOptimizer:
            def __init__(self, f, constraints):
                self.f = f
                self.constraints = constraints

            def barrier_function_method(self, x0, mu=10, tolerance=1e-6):
                """
                Método da função barreira
                """
                x_current = x0
                t = 1.0  # Parâmetro da barreira

                while True:
                    # Função barreira modificada
                    def modified_objective(x):
                        phi = self.f(x)
                        for constraint in self.constraints:
                            g = constraint['function'](x)
                            if g >= 0:  # Violação de restrição
                                return float('inf')
                            phi -= (1/t) * np.log(-g)  # Função barreira

                        return phi

                    # Minimizações sucessivas
                    result = minimize_scalar(
                        lambda x_val: modified_objective(np.array([x_val])),
                        bounds=(-10, 10),
                        method='bounded'
                    )

                    x_new = np.array([result.x])

                    # Verificar convergência
                    if np.linalg.norm(x_new - x_current) < tolerance:
                        break

                    x_current = x_new
                    t *= mu  # Aumentar parâmetro

                return x_current

        return InteriorPointOptimizer(objective_function, constraints)

    def nonconvex_optimization_challenges(self, function, domain):
        """
        Desafios da otimização não-convexa
        """
        class NonConvexAnalysis:
            def __init__(self, f, domain):
                self.f = f
                self.domain = domain

            def identify_local_minima(self, n_starts=10):
                """Identifica mínimos locais usando múltiplas inicializações"""
                local_minima = []

                for _ in range(n_starts):
                    x0 = np.random.uniform(self.domain[0], self.domain[1])

                    # Otimização local
                    result = minimize_scalar(
                        self.f,
                        bounds=self.domain,
                        method='bounded'
                    )

                    local_minima.append({
                        'x': result.x,
                        'f_value': result.fun,
                        'success': result.success
                    })

                # Encontrar mínimo global aproximado
                best_minimum = min(local_minima, key=lambda x: x['f_value'])

                return {
                    'local_minima': local_minima,
                    'global_minimum_approx': best_minimum,
                    'n_local_minima': len(set(round(m['f_value'], 3) for m in local_minima))
                }

            def landscape_analysis(self, resolution=100):
                """Análise da paisagem da função"""
                x_values = np.linspace(self.domain[0], self.domain[1], resolution)
                y_values = [self.f(x) for x in x_values]

                # Calcular curvatura local
                curvature = []
                for i in range(1, len(y_values) - 1):
                    second_derivative = (y_values[i+1] - 2*y_values[i] + y_values[i-1]) / (
                        (x_values[i+1] - x_values[i]) ** 2
                    )
                    curvature.append(second_derivative)

                return {
                    'x_values': x_values,
                    'y_values': y_values,
                    'curvature': curvature,
                    'convex_regions': sum(1 for c in curvature if c > 0),
                    'concave_regions': sum(1 for c in curvature if c < 0)
                }

        return NonConvexAnalysis(function, domain)

    def stochastic_gradient_methods(self, objective_function, data_generator):
        """
        Métodos de gradiente estocástico
        """
        class StochasticGradientDescent:
            def __init__(self, f, data_gen):
                self.f = f
                self.data_generator = data_gen

            def sgd_optimization(self, x0, learning_rate=0.01, n_iterations=1000):
                """Gradiente descendente estocástico básico"""
                x_current = x0
                trajectory = [x_current]

                for iteration in range(n_iterations):
                    # Amostra de dados
                    data_sample = next(self.data_generator)

                    # Gradiente estocástico
                    stochastic_grad = self._stochastic_gradient(x_current, data_sample)

                    # Atualização
                    x_new = x_current - learning_rate * stochastic_grad
                    x_current = x_new

                    trajectory.append(x_current)

                    # Decaimento da taxa de aprendizado
                    learning_rate *= 0.99

                return {
                    'final_solution': x_current,
                    'trajectory': trajectory,
                    'convergence_metric': np.var(trajectory[-10:])  # Variação recente
                }

            def adam_optimizer(self, x0, alpha=0.001, beta1=0.9, beta2=0.999):
                """Otimizador Adam"""
                x_current = x0
                m = 0  # Primeiro momento
                v = 0  # Segundo momento
                t = 0

                for iteration in range(1000):
                    t += 1
                    data_sample = next(self.data_generator)

                    # Gradiente estocástico
                    g = self._stochastic_gradient(x_current, data_sample)

                    # Atualizar momentos
                    m = beta1 * m + (1 - beta1) * g
                    v = beta2 * v + (1 - beta2) * (g ** 2)

                    # Correção de bias
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)

                    # Atualização
                    x_new = x_current - alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
                    x_current = x_new

                return x_current

            def _stochastic_gradient(self, x, data_sample):
                """Calcula gradiente estocástico"""
                # Implementação simplificada
                h = 1e-5
                f_plus = self.f(x + h, data_sample)
                f_minus = self.f(x - h, data_sample)

                return (f_plus - f_minus) / (2 * h)

        return StochasticGradientDescent(objective_function, data_generator)
```

**Conceitos Críticos:**
- Funções convexas e propriedades
- Dualidade Lagrangiana e condições KKT
- Métodos de pontos interiores
- Otimização não-convexa e paisagem de funções
- Métodos de gradiente estocástico (SGD, Adam)

### 1.3 Teoria da Informação e Complexidade
```python
import numpy as np
from scipy.stats import entropy
import math

class InformationTheoryML:
    """
    Teoria da informação aplicada ao aprendizado de máquina
    """

    def __init__(self):
        self.entropy_cache = {}

    def kullback_leibler_divergence(self, p, q):
        """
        Divergência KL: medida de diferença entre distribuições
        D_KL(p||q) = Σ p(x) log(p(x)/q(x))
        """
        # Evitar divisão por zero
        q_safe = np.where(q == 0, 1e-10, q)

        kl_div = np.sum(p * np.log(p / q_safe))

        return kl_div

    def mutual_information(self, joint_distribution, marginal_x, marginal_y):
        """
        Informação mútua: I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        # Entropias
        H_X = entropy(marginal_x)
        H_Y = entropy(marginal_y)
        H_XY = entropy(joint_distribution.flatten())

        mutual_info = H_X + H_Y - H_XY

        return mutual_info

    def information_bottleneck_method(self, X, Y, compression_rate=0.5):
        """
        Método do gargalo de informação para aprendizado representacional
        """
        class InformationBottleneck:
            def __init__(self, X, Y, beta):
                self.X = X
                self.Y = Y
                self.beta = beta  # Parâmetro de compressão

            def optimize_representation(self, n_iterations=100):
                """Otimiza representação comprimida"""
                # Inicialização aleatória da representação
                T = np.random.randn(len(self.X), int(len(self.X[0]) * compression_rate))

                for iteration in range(n_iterations):
                    # Passo E: Atualizar distribuição p(t|x)
                    p_t_given_x = self._update_encoder(T)

                    # Passo M: Atualizar representação T
                    T = self._update_representation(p_t_given_x)

                return T

            def _update_encoder(self, T):
                """Atualiza codificador p(t|x)"""
                # Implementação simplificada
                return np.random.rand(len(self.X), len(T[0]))

            def _update_representation(self, p_t_given_x):
                """Atualiza representação T"""
                # Implementação simplificada
                return np.random.randn(len(self.X), len(p_t_given_x[0]))

            def calculate_ib_objective(self, T, p_t_given_x):
                """Calcula função objetivo do IB"""
                # I(Y;T) - β I(X;T)
                I_YT = self._mutual_information_YT(T)
                I_XT = self._mutual_information_XT(T, p_t_given_x)

                objective = I_YT - self.beta * I_XT

                return objective

            def _mutual_information_YT(self, T):
                """I(Y;T)"""
                # Implementação simplificada
                return np.random.rand()

            def _mutual_information_XT(self, T, p_t_given_x):
                """I(X;T)"""
                # Implementação simplificada
                return np.random.rand()

        return InformationBottleneck(X, Y, compression_rate)

    def minimum_description_length(self, model, data):
        """
        Princípio do comprimento mínimo de descrição (MDL)
        """
        class MDLPrinciple:
            def __init__(self, model, data):
                self.model = model
                self.data = data

            def calculate_mdl_score(self):
                """Calcula score MDL = L(D|M) + L(M)"""
                # Comprimento de descrição dos dados dado o modelo
                data_description_length = self._data_description_length()

                # Comprimento de descrição do modelo
                model_description_length = self._model_description_length()

                total_mdl = data_description_length + model_description_length

                return total_mdl

            def _data_description_length(self):
                """L(D|M) - comprimento para codificar dados"""
                # Usando entropia do modelo
                if hasattr(self.model, 'predict_proba'):
                    # Modelo probabilístico
                    probs = self.model.predict_proba(self.data)
                    data_entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    return data_entropy
                else:
                    # Modelo determinístico
                    return len(self.data) * np.log2(2)  # Codificação binária simples

            def _model_description_length(self):
                """L(M) - complexidade do modelo"""
                if hasattr(self.model, 'get_params'):
                    params = self.model.get_params()
                    # Número de parâmetros como medida de complexidade
                    complexity = len(params)
                    return complexity * np.log2(len(self.data))
                else:
                    return 0

            def model_selection_mdl(self, candidate_models):
                """Seleção de modelo usando MDL"""
                best_model = None
                best_mdl = float('inf')

                for model in candidate_models:
                    mdl_calculator = MDLPrinciple(model, self.data)
                    mdl_score = mdl_calculator.calculate_mdl_score()

                    if mdl_score < best_mdl:
                        best_mdl = mdl_score
                        best_model = model

                return best_model, best_mdl

        return MDLPrinciple(model, data)

    def algorithmic_complexity(self, data_sequence):
        """
        Complexidade algorítmica (Kolmogorov)
        """
        class KolmogorovComplexity:
            def __init__(self, sequence):
                self.sequence = sequence

            def approximate_k_complexity(self):
                """Aproximação da complexidade de Kolmogorov"""
                # Usar compressão como proxy
                import zlib

                # Complexidade = tamanho da sequência comprimida
                compressed = zlib.compress(str(self.sequence).encode())
                k_complexity = len(compressed)

                # Normalizar pelo tamanho original
                original_size = len(str(self.sequence).encode())
                normalized_complexity = k_complexity / original_size

                return {
                    'kolmogorov_complexity': k_complexity,
                    'normalized_complexity': normalized_complexity,
                    'compression_ratio': normalized_complexity
                }

            def randomness_test(self):
                """Teste de aleatoriedade baseado em complexidade"""
                complexity_result = self.approximate_k_complexity()

                # Sequências aleatórias têm alta complexidade
                # Sequências com padrão têm baixa complexity
                is_random = complexity_result['normalized_complexity'] > 0.8

                return {
                    'is_random': is_random,
                    'complexity_score': complexity_result['normalized_complexity']
                }

        return KolmogorovComplexity(data_sequence)

    def rate_distortion_theory(self, source_distribution, distortion_measure):
        """
        Teoria da taxa-distorção para compressão de dados
        """
        class RateDistortionAnalysis:
            def __init__(self, source, distortion):
                self.source = source
                self.distortion = distortion

            def calculate_rate_distortion_function(self, distortion_levels):
                """Calcula função taxa-distorção R(D)"""
                rate_distortion_curve = []

                for D in distortion_levels:
                    # Taxa mínima necessária para alcançar distorção D
                    min_rate = self._find_minimum_rate(D)
                    rate_distortion_curve.append((D, min_rate))

                return rate_distortion_curve

            def _find_minimum_rate(self, target_distortion):
                """Encontra taxa mínima para distorção alvo"""
                # Implementação simplificada usando entropia
                source_entropy = entropy(self.source)

                # Aproximação: R(D) ≈ H(X) - log2(1/D) para D pequeno
                if target_distortion < 0.1:
                    min_rate = max(0, source_entropy - np.log2(1/target_distortion))
                else:
                    min_rate = source_entropy * 0.5  # Aproximação

                return min_rate

            def distortion_rate_function(self, rate):
                """Calcula D(R) - distorção para taxa dada"""
                # Função inversa
                if rate >= self.source.max():
                    return 0  # Sem distorção
                else:
                    # Aproximação exponencial
                    return np.exp(-rate / entropy(self.source))

        return RateDistortionAnalysis(source_distribution, distortion_measure)

    def information_geometry_ml(self, parameter_space):
        """
        Geometria da informação em aprendizado de máquina
        """
        class InformationGeometry:
            def __init__(self, parameter_space):
                self.parameter_space = parameter_space

            def fisher_information_matrix(self, log_likelihood_function):
                """Matriz de informação de Fisher"""
                n_params = len(self.parameter_space)

                # Inicializar matriz
                FIM = np.zeros((n_params, n_params))

                # Calcular derivadas segundas da log-verossimilhança
                for i in range(n_params):
                    for j in range(n_params):
                        # E[-d²logL/dθi dθj]
                        second_derivative = self._numerical_second_derivative(
                            log_likelihood_function, i, j
                        )
                        FIM[i, j] = -second_derivative

                return FIM

            def _numerical_second_derivative(self, f, i, j, h=1e-5):
                """Derivada segunda numérica"""
                theta = np.array(self.parameter_space)

                # Derivadas parciais
                def partial_i(f_val):
                    theta_plus = theta.copy()
                    theta_plus[i] += h
                    theta_minus = theta.copy()
                    theta_minus[i] -= h

                    return (f(theta_plus) - f(theta_minus)) / (2 * h)

                # Derivada segunda
                partial_i_at_plus = partial_i(lambda t: partial_i(lambda s: f(s) if s[j] == theta[j] else f(np.where(np.arange(len(theta)) == j, theta[j] + h, theta))))
                partial_i_at_minus = partial_i(lambda t: partial_i(lambda s: f(s) if s[j] == theta[j] else f(np.where(np.arange(len(theta)) == j, theta[j] - h, theta))))

                return (partial_i_at_plus - partial_i_at_minus) / (2 * h)

            def natural_gradient(self, gradient, fim):
                """Gradiente natural usando informação de Fisher"""
                # g_natural = F^{-1} g_ordinário
                fim_inv = np.linalg.inv(fim)
                natural_grad = np.dot(fim_inv, gradient)

                return natural_grad

            def amari_distance(self, distribution1, distribution2):
                """Distância de Amari entre distribuições"""
                # Distância de informação de Fisher-Rao
                # Implementação simplificada
                kl_12 = self.kullback_leibler_divergence(distribution1, distribution2)
                kl_21 = self.kullback_leibler_divergergence(distribution2, distribution1)

                amari_dist = (kl_12 + kl_21) / 2

                return amari_dist

        return InformationGeometry(parameter_space)
```

**Conceitos Críticos:**
- Divergência KL e informação mútua
- Gargalo de informação
- Princípio do comprimento mínimo de descrição
- Complexidade algorítmica de Kolmogorov
- Geometria da informação e gradiente natural

---

## 2. APRENDIZADO ESTATÍSTICO E TEORIA PAC

### 2.1 Aprendizado Supervisionado: VC-Dimensão e Rademacher
```python
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

class StatisticalLearningTheory:
    """
    Teoria do aprendizado estatístico - VC-dimension e complexidade
    """

    def __init__(self):
        self.vc_dimension_cache = {}

    def vc_dimension_calculation(self, hypothesis_class, data_points):
        """
        Cálculo da dimensão VC de uma classe de hipóteses
        """
        class VC_Dimension_Analyzer:
            def __init__(self, hypothesis_class):
                self.hypothesis_class = hypothesis_class

            def shatter_test(self, points):
                """Testa se os pontos podem ser shattered"""
                n_points = len(points)

                # Gerar todas as dicotomias possíveis
                all_labels = []
                for i in range(2**n_points):
                    labels = [(i >> j) & 1 for j in range(n_points)]
                    all_labels.append(labels)

                shattered_count = 0

                for labels in all_labels:
                    # Verificar se existe hipótese que produz estes rótulos
                    can_shatter = False

                    for hypothesis in self.hypothesis_class:
                        predicted_labels = [hypothesis(point) for point in points]

                        if predicted_labels == labels:
                            can_shatter = True
                            break

                    if can_shatter:
                        shattered_count += 1

                # Se todas as dicotomias são possíveis, os pontos são shattered
                return shattered_count == 2**n_points

            def calculate_vc_dimension(self, max_test_points=10):
                """Calcula dimensão VC"""
                vc_dim = 0

                for d in range(1, max_test_points + 1):
                    # Gerar pontos de teste
                    test_points = np.random.randn(d, 2)  # Pontos 2D

                    if self.shatter_test(test_points):
                        vc_dim = d
                    else:
                        break

                return vc_dim

            def sauers_lemma_bound(self, n_samples, delta=0.05):
                """Limite de Sauer para crescimento da classe de hipóteses"""
                vc_dim = self.calculate_vc_dimension()

                if vc_dim is None:
                    return float('inf')

                # |H| ≤ Σ_{i=0}^{VC} C(n, i)
                growth_function = sum(comb(n_samples, i) for i in range(vc_dim + 1))

                return growth_function

        return VC_Dimension_Analyzer(hypothesis_class)

    def rademacher_complexity_analysis(self, hypothesis_class, data_distribution):
        """
        Análise de complexidade de Rademacher
        """
        class RademacherComplexity:
            def __init__(self, hypothesis_class, data_dist):
                self.hypothesis_class = hypothesis_class
                self.data_distribution = data_dist

            def empirical_rademacher_complexity(self, sample_size, n_experiments=100):
                """Complexidade de Rademacher empírica"""
                rademacher_scores = []

                for _ in range(n_experiments):
                    # Gerar amostra
                    sample = self.data_distribution.rvs(size=sample_size)

                    # Vetor de Rademacher
                    sigma = np.random.choice([-1, 1], size=sample_size)

                    # Calcular complexidade
                    max_correlation = 0

                    for hypothesis in self.hypothesis_class:
                        predictions = [hypothesis(x) for x in sample]
                        correlation = np.mean(sigma * predictions)
                        max_correlation = max(max_correlation, abs(correlation))

                    rademacher_scores.append(max_correlation)

                empirical_rad = np.mean(rademacher_scores)

                return empirical_rad

            def generalization_bound_rademacher(self, empirical_risk, sample_size, confidence=0.95):
                """Limite de generalização usando complexidade de Rademacher"""
                empirical_rad = self.empirical_rademacher_complexity(sample_size)

                # Desvio padrão das variáveis de Rademacher
                rad_std = np.sqrt(np.log(2 / (1 - confidence)) / (2 * sample_size))

                generalization_error = empirical_risk + 2 * empirical_rad + rad_std

                return generalization_error

        return RademacherComplexity(hypothesis_class, data_distribution)

    def uniform_convergence_theory(self, hypothesis_class, loss_function):
        """
        Teoria da convergência uniforme
        """
        class UniformConvergence:
            def __init__(self, hypothesis_class, loss_func):
                self.hypothesis_class = hypothesis_class
                self.loss_function = loss_func

            def symmetrization_inequality(self, sample_size, epsilon=0.1):
                """Desigualdade de simetrização de Vapnik-Chervonenkis"""
                # P(sup |L_D(h) - L(h)| > ε) ≤ 2 * N_H(2n) * exp(-nε²/2)

                # Número de hipóteses
                N = len(self.hypothesis_class) if hasattr(self.hypothesis_class, '__len__') else 1000

                # Probabilidade
                prob_bound = 2 * N * np.exp(-sample_size * epsilon**2 / 2)

                return prob_bound

            def generalization_error_bound(self, empirical_error, sample_size, vc_dim=None):
                """Limite para erro de generalização"""
                if vc_dim is None:
                    # Usar limite genérico
                    vc_dim = 10  # Assumir

                # sqrt( (VC log(2n/VC) + log(1/δ) ) / n )
                delta = 0.05
                sqrt_term = np.sqrt((vc_dim * np.log(2 * sample_size / vc_dim) + np.log(1/delta)) / sample_size)

                generalization_bound = empirical_error + sqrt_term

                return generalization_bound

        return UniformConvergence(hypothesis_class, loss_function)

    def bias_variance_decomposition(self, true_function, hypothesis_class, data_generator):
        """
        Decomposição bias-variância do erro
        """
        class BiasVarianceAnalysis:
            def __init__(self, true_func, hypothesis_class, data_gen):
                self.true_function = true_func
                self.hypothesis_class = hypothesis_class
                self.data_generator = data_gen

            def decompose_error(self, test_point, n_experiments=100, sample_sizes=[10, 50, 100, 500]):
                """Decompõe erro em bias² + variância + ruído"""
                error_decomposition = {}

                for n in sample_sizes:
                    biases = []
                    variances = []
                    noises = []

                    for _ in range(n_experiments):
                        # Gerar dados de treinamento
                        X_train, y_train = self._generate_training_data(n)

                        # Treinar hipótese
                        hypothesis = self._train_hypothesis(X_train, y_train)

                        # Avaliar no ponto de teste
                        prediction = hypothesis(test_point)
                        true_value = self.true_function(test_point)

                        # Calcular componentes
                        bias = prediction - true_value
                        noise = np.random.normal(0, 0.1)  # Ruído assumido

                        biases.append(bias**2)
                        variances.append((prediction - np.mean([h(test_point) for h in self.hypothesis_class]))**2)
                        noises.append(noise**2)

                    # Médias
                    avg_bias_squared = np.mean(biases)
                    avg_variance = np.mean(variances)
                    avg_noise = np.mean(noises)

                    error_decomposition[n] = {
                        'bias_squared': avg_bias_squared,
                        'variance': avg_variance,
                        'noise': avg_noise,
                        'total_error': avg_bias_squared + avg_variance + avg_noise
                    }

                return error_decomposition

            def _generate_training_data(self, n):
                """Gera dados de treinamento"""
                X = np.random.uniform(-1, 1, n)
                y = [self.true_function(x) + np.random.normal(0, 0.1) for x in X]
                return X, y

            def _train_hypothesis(self, X, y):
                """Treina hipótese simples (regressão linear)"""
                # Ajuste linear simples
                X_mean = np.mean(X)
                y_mean = np.mean(y)

                numerator = np.sum((X - X_mean) * (y - y_mean))
                denominator = np.sum((X - X_mean)**2)

                if denominator == 0:
                    slope = 0
                else:
                    slope = numerator / denominator

                intercept = y_mean - slope * X_mean

                def linear_hypothesis(x):
                    return slope * x + intercept

                return linear_hypothesis

        return BiasVarianceAnalysis(true_function, hypothesis_class, data_generator)

    def pac_learning_framework(self, concept_class, hypothesis_class):
        """
        Framework de aprendizado PAC (Probably Approximately Correct)
        """
        class PAC_Learning:
            def __init__(self, concept_class, hypothesis_class):
                self.concept_class = concept_class
                self.hypothesis_class = hypothesis_class

            def sample_complexity_bound(self, accuracy, confidence):
                """Limite de complexidade de amostra"""
                # Para classes agnostically learnable
                # m ≥ (1/ε) * (VC_dim * log(1/δ) + log(1/δ))

                vc_dim = 10  # Assumir
                epsilon = 1 - accuracy
                delta = 1 - confidence

                sample_complexity = (1/epsilon) * (vc_dim * np.log(1/delta) + np.log(1/delta))

                return int(sample_complexity)

            def learnability_test(self):
                """Testa se a classe é PAC-learnable"""
                # Critérios para PAC-learnabilidade
                criteria = {
                    'concept_class_consistent': self._check_concept_consistency(),
                    'hypothesis_class_sufficient': self._check_hypothesis_sufficiency(),
                    'polynomial_sample_complexity': self._check_polynomial_complexity()
                }

                is_pac_learnable = all(criteria.values())

                return {
                    'is_pac_learnable': is_pac_learnable,
                    'criteria': criteria
                }

            def _check_concept_consistency(self):
                """Verifica consistência da classe de conceitos"""
                # Implementação simplificada
                return True

            def _check_hypothesis_sufficiency(self):
                """Verifica suficiência da classe de hipóteses"""
                # Implementação simplificada
                return True

            def _check_polynomial_complexity(self):
                """Verifica complexidade polinomial"""
                # Implementação simplificada
                return True

        return PAC_Learning(concept_class, hypothesis_class)
```

**Conceitos Críticos:**
- Dimensão VC e crescimento de classes de hipóteses
- Complexidade de Rademacher
- Teoria da convergência uniforme
- Decomposição bias-variância
- Framework PAC de aprendizado

### 2.2 Aprendizado Não-Supervisionado: Misturas e EM
```python
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class UnsupervisedLearningTheory:
    """
    Teoria do aprendizado não-supervisionado
    """

    def __init__(self):
        self.mixture_models = {}

    def gaussian_mixture_models(self, data, n_components=3):
        """
        Modelos de mistura Gaussiana e algoritmo EM
        """
        class GaussianMixtureEM:
            def __init__(self, data, n_components):
                self.data = data
                self.n_components = n_components
                self.n_features = data.shape[1] if len(data.shape) > 1 else 1
                self.n_samples = len(data)

                # Inicializar parâmetros
                self.weights = np.ones(n_components) / n_components
                self.means = self._initialize_means()
                self.covariances = self._initialize_covariances()

            def _initialize_means(self):
                """Inicializa médias usando K-means"""
                if self.n_features == 1:
                    # 1D: usar quantis
                    quantiles = np.linspace(0.1, 0.9, self.n_components)
                    means = np.quantile(self.data, quantiles)
                else:
                    # Multidimensional: usar K-means
                    kmeans = KMeans(n_clusters=self.n_components, random_state=42)
                    kmeans.fit(self.data)
                    means = kmeans.cluster_centers_

                return means

            def _initialize_covariances(self):
                """Inicializa matrizes de covariância"""
                if self.n_features == 1:
                    # 1D: variância simples
                    covs = np.array([np.var(self.data)] * self.n_components)
                else:
                    # Multidimensional: covariância da amostra
                    covs = np.array([np.cov(self.data.T)] * self.n_components)

                return covs

            def expectation_maximization(self, max_iterations=100, tolerance=1e-6):
                """Algoritmo EM para GMM"""
                log_likelihood_history = []

                for iteration in range(max_iterations):
                    # Passo E: Calcular responsabilidades
                    responsibilities = self._e_step()

                    # Passo M: Atualizar parâmetros
                    self._m_step(responsibilities)

                    # Calcular log-verossimilhança
                    log_likelihood = self._calculate_log_likelihood()
                    log_likelihood_history.append(log_likelihood)

                    # Verificar convergência
                    if iteration > 1:
                        if abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < tolerance:
                            break

                return {
                    'weights': self.weights,
                    'means': self.means,
                    'covariances': self.covariances,
                    'log_likelihood_history': log_likelihood_history,
                    'responsibilities': responsibilities,
                    'converged': iteration < max_iterations
                }

            def _e_step(self):
                """Passo de Expectation"""
                responsibilities = np.zeros((self.n_samples, self.n_components))

                for i in range(self.n_samples):
                    for k in range(self.n_components):
                        # Probabilidade de pertencer ao componente k
                        if self.n_features == 1:
                            # 1D Gaussian
                            likelihood = self._gaussian_pdf_1d(self.data[i], self.means[k], self.covariances[k])
                        else:
                            # Multidimensional Gaussian
                            likelihood = multivariate_normal.pdf(
                                self.data[i], self.means[k], self.covariances[k]
                            )

                        responsibilities[i, k] = self.weights[k] * likelihood

                    # Normalizar
                    if np.sum(responsibilities[i, :]) > 0:
                        responsibilities[i, :] /= np.sum(responsibilities[i, :])

                return responsibilities

            def _m_step(self, responsibilities):
                """Passo de Maximização"""
                # Atualizar pesos
                N_k = np.sum(responsibilities, axis=0)
                self.weights = N_k / self.n_samples

                # Atualizar médias
                for k in range(self.n_components):
                    if N_k[k] > 0:
                        weighted_sum = np.sum(
                            responsibilities[:, k].reshape(-1, 1) * self.data,
                            axis=0
                        )
                        self.means[k] = weighted_sum / N_k[k]

                # Atualizar covariâncias
                for k in range(self.n_components):
                    if N_k[k] > 0:
                        diff = self.data - self.means[k]
                        weighted_cov = np.zeros((self.n_features, self.n_features))

                        for i in range(self.n_samples):
                            outer_prod = np.outer(diff[i], diff[i])
                            weighted_cov += responsibilities[i, k] * outer_prod

                        self.covariances[k] = weighted_cov / N_k[k]

                        # Regularização para evitar singularidade
                        self.covariances[k] += 1e-6 * np.eye(self.n_features)

            def _calculate_log_likelihood(self):
                """Calcula log-verossimilhança"""
                log_likelihood = 0

                for i in range(self.n_samples):
                    sample_likelihood = 0

                    for k in range(self.n_components):
                        if self.n_features == 1:
                            likelihood = self._gaussian_pdf_1d(self.data[i], self.means[k], self.covariances[k])
                        else:
                            likelihood = multivariate_normal.pdf(
                                self.data[i], self.means[k], self.covariances[k]
                            )

                        sample_likelihood += self.weights[k] * likelihood

                    if sample_likelihood > 0:
                        log_likelihood += np.log(sample_likelihood)

                return log_likelihood

            def _gaussian_pdf_1d(self, x, mean, variance):
                """PDF Gaussiana 1D"""
                return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(
                    -0.5 * ((x - mean) ** 2) / variance
                )

            def predict_cluster_probabilities(self, new_data):
                """Prevê probabilidades de cluster para novos dados"""
                n_new_samples = len(new_data)
                probabilities = np.zeros((n_new_samples, self.n_components))

                for i in range(n_new_samples):
                    for k in range(self.n_components):
                        if self.n_features == 1:
                            likelihood = self._gaussian_pdf_1d(new_data[i], self.means[k], self.covariances[k])
                        else:
                            likelihood = multivariate_normal.pdf(
                                new_data[i], self.means[k], self.covariances[k]
                            )

                        probabilities[i, k] = self.weights[k] * likelihood

                    # Normalizar
                    if np.sum(probabilities[i, :]) > 0:
                        probabilities[i, :] /= np.sum(probabilities[i, :])

                return probabilities

        return GaussianMixtureEM(data, n_components)

    def latent_variable_models(self, observed_data, latent_dimension):
        """
        Modelos com variáveis latentes
        """
        class LatentVariableModel:
            def __init__(self, observed_data, latent_dim):
                self.observed_data = observed_data
                self.latent_dimension = latent_dim
                self.n_samples, self.observed_dimension = observed_data.shape

            def principal_component_analysis(self):
                """Análise de componentes principais (PCA)"""
                # Centralizar dados
                data_centered = self.observed_data - np.mean(self.observed_data, axis=0)

                # Matriz de covariância
                cov_matrix = np.cov(data_centered.T)

                # Autovalores e autovetores
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                # Ordenar em ordem decrescente
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # Componentes principais
                principal_components = eigenvectors[:, :self.latent_dimension]

                # Projeção dos dados
                projected_data = np.dot(data_centered, principal_components)

                # Variância explicada
                explained_variance = eigenvalues[:self.latent_dimension] / np.sum(eigenvalues)

                return {
                    'principal_components': principal_components,
                    'projected_data': projected_data,
                    'explained_variance': explained_variance,
                    'eigenvalues': eigenvalues,
                    'eigenvectors': eigenvectors
                }

            def factor_analysis(self):
                """Análise fatorial"""
                # Modelo: X = ΛF + ε
                # Onde Λ é matriz de loadings, F são fatores, ε é erro

                # Implementação simplificada
                n_factors = self.latent_dimension

                # Inicializar loadings
                loadings = np.random.randn(self.observed_dimension, n_factors)

                # Algoritmo EM simplificado para análise fatorial
                max_iter = 50
                for iteration in range(max_iter):
                    # Passo E: Estimar fatores
                    factors = self._estimate_factors(loadings)

                    # Passo M: Atualizar loadings
                    loadings = self._update_loadings(factors)

                # Calcular unicidades (variâncias específicas)
                uniquenesses = self._calculate_uniquenesses(loadings, factors)

                return {
                    'loadings': loadings,
                    'factors': factors,
                    'uniquenesses': uniquenesses
                }

            def _estimate_factors(self, loadings):
                """Estima fatores latentes"""
                # Solução de mínimos quadrados
                factors = np.linalg.lstsq(loadings, self.observed_data.T, rcond=None)[0].T
                return factors

            def _update_loadings(self, factors):
                """Atualiza matriz de loadings"""
                loadings = np.dot(self.observed_data.T, factors)
                loadings = loadings / np.linalg.norm(loadings, axis=0)
                return loadings

            def _calculate_uniquenesses(self, loadings, factors):
                """Calcula unicidades (variâncias específicas)"""
                # Ψ = diag(Σ - ΛΛ^T)
                reconstructed_cov = np.dot(loadings, loadings.T)
                sample_cov = np.cov(self.observed_data.T)

                uniquenesses = np.diag(sample_cov - reconstructed_cov)
                uniquenesses = np.maximum(uniquenesses, 0)  # Não negativo

                return uniquenesses

        return LatentVariableModel(observed_data, latent_dimension)

    def spectral_clustering_theory(self, similarity_matrix):
        """
        Teoria do agrupamento espectral
        """
        class SpectralClustering:
            def __init__(self, similarity_matrix):
                self.W = similarity_matrix
                self.n_points = similarity_matrix.shape[0]

                # Matriz de grau
                self.D = np.diag(np.sum(self.W, axis=1))

                # Laplaciano
                self.L = self.D - self.W

                # Laplaciano normalizado
                self.D_inv_sqrt = np.linalg.inv(np.sqrt(self.D))
                self.L_norm = self.D_inv_sqrt @ self.L @ self.D_inv_sqrt

            def compute_eigenvectors(self, n_clusters):
                """Computa autovetores do Laplaciano"""
                # Autovalores e autovetores do Laplaciano normalizado
                eigenvalues, eigenvectors = np.linalg.eigh(self.L_norm)

                # Selecionar primeiros autovetores (ignorando primeiro autovalor ≈ 0)
                selected_eigenvectors = eigenvectors[:, 1:n_clusters+1]

                return selected_eigenvectors, eigenvalues[1:n_clusters+1]

            def cluster_eigenvectors(self, eigenvectors, n_clusters):
                """Agrupa os autovetores usando K-means"""
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(eigenvectors)

                return labels

            def spectral_clustering_algorithm(self, n_clusters):
                """Algoritmo completo de agrupamento espectral"""
                # Passo 1: Computar autovetores
                eigenvectors, eigenvalues = self.compute_eigenvectors(n_clusters)

                # Passo 2: Formar matriz de características
                feature_matrix = eigenvectors

                # Passo 3: Agrupar
                labels = self.cluster_eigenvectors(feature_matrix, n_clusters)

                return {
                    'labels': labels,
                    'eigenvectors': eigenvectors,
                    'eigenvalues': eigenvalues,
                    'feature_matrix': feature_matrix
                }

            def cheeger_constant(self):
                """Constante de Cheeger (medida de conectividade)"""
                # h(G) = min_{S ⊂ V, |S| ≤ n/2} |E(S, V\S)| / min(|S|, |V\S|)

                # Implementação simplificada: usar segundo menor autovalor
                eigenvalues, _ = np.linalg.eigh(self.L)
                cheeger_constant = eigenvalues[1] / 2

                return cheeger_constant

        return SpectralClustering(similarity_matrix)

    def manifold_learning_theory(self, high_dim_data, target_dimension):
        """
        Aprendizado de variedades
        """
        class ManifoldLearning:
            def __init__(self, data, target_dim):
                self.data = data
                self.target_dimension = target_dim
                self.n_samples, self.n_features = data.shape

            def locally_linear_embedding(self, n_neighbors=10):
                """Incorporação linear local (LLE)"""
                # Passo 1: Encontrar vizinhos
                neighbors = self._find_neighbors(n_neighbors)

                # Passo 2: Computar pesos de reconstrução
                reconstruction_weights = self._compute_reconstruction_weights(neighbors)

                # Passo 3: Computar incorporação de baixa dimensão
                embedding = self._compute_low_dim_embedding(reconstruction_weights)

                return embedding

            def _find_neighbors(self, n_neighbors):
                """Encontra vizinhos mais próximos"""
                from sklearn.neighbors import NearestNeighbors

                nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(self.data)
                distances, indices = nbrs.kneighbors(self.data)

                return indices

            def _compute_reconstruction_weights(self, neighbors):
                """Computa pesos de reconstrução"""
                weights = np.zeros((self.n_samples, self.n_samples))

                for i in range(self.n_samples):
                    # Pontos vizinhos
                    neighbor_indices = neighbors[i]
                    neighbor_points = self.data[neighbor_indices]

                    # Centro nos vizinhos
                    neighbor_points = neighbor_points - self.data[i]

                    # Resolver sistema linear para pesos
                    # Minimizar ||X W - X_0|| sujeito a ΣW = 1
                    X = neighbor_points.T
                    ones = np.ones(len(neighbor_indices))

                    # Sistema: [X; ones.T] W = [zeros; 1]
                    A = np.vstack([X, ones])
                    b = np.zeros(self.n_features + 1)
                    b[-1] = 1

                    # Solução por mínimos quadrados regularizados
                    W = np.linalg.lstsq(A.T @ A + 1e-3 * np.eye(A.shape[1]),
                                       A.T @ b, rcond=None)[0]

                    # Atribuir pesos
                    for j, neighbor_idx in enumerate(neighbor_indices):
                        weights[i, neighbor_idx] = W[j]

                return weights

            def _compute_low_dim_embedding(self, weights):
                """Computa incorporação de baixa dimensão"""
                # Resolver problema de autovalores
                # Minimizar Σ ||Y_i - Σ W_ij Y_j||² sujeito a Y^T Y = I

                # Matriz de pesos
                W = weights

                # Matriz M = (I - W)^T (I - W)
                I = np.eye(self.n_samples)
                M = (I - W).T @ (I - W)

                # Autovalores e autovetores
                eigenvalues, eigenvectors = np.linalg.eigh(M)

                # Selecionar autovetores correspondentes aos menores autovalores
                idx = np.argsort(eigenvalues)
                embedding = eigenvectors[:, idx[1:self.target_dimension+1]]  # Ignorar primeiro

                return embedding

            def isomap_algorithm(self, n_neighbors=10):
                """Isomap: Isometric Feature Mapping"""
                # Passo 1: Construir grafo de vizinhança
                distances = self._compute_geodesic_distances(n_neighbors)

                # Passo 2: Aplicar MDS (Multidimensional Scaling)
                embedding = self._classical_mds(distances)

                return embedding

            def _compute_geodesic_distances(self, n_neighbors):
                """Computa distâncias geodésicas aproximadas"""
                from sklearn.neighbors import kneighbors_graph
                from scipy.sparse.csgraph import shortest_path

                # Grafo de vizinhança
                graph = kneighbors_graph(self.data, n_neighbors, mode='distance')

                # Distâncias geodésicas (caminhos mais curtos)
                geodesic_distances = shortest_path(graph, directed=False)

                return geodesic_distances

            def _classical_mds(self, distance_matrix):
                """Escalonamento multidimensional clássico"""
                # Matriz B = -0.5 * J * D² * J
                n = distance_matrix.shape[0]
                J = np.eye(n) - np.ones((n, n)) / n
                B = -0.5 * J @ (distance_matrix ** 2) @ J

                # Autovalores e autovetores
                eigenvalues, eigenvectors = np.linalg.eigh(B)

                # Selecionar componentes positivas
                positive_idx = eigenvalues > 0
                eigenvalues_pos = eigenvalues[positive_idx]
                eigenvectors_pos = eigenvectors[:, positive_idx]

                # Incorporação
                embedding = eigenvectors_pos[:, :self.target_dimension] @ np.diag(
                    np.sqrt(eigenvalues_pos[:self.target_dimension])
                )

                return embedding

        return ManifoldLearning(high_dim_data, target_dimension)
```

**Conceitos Críticos:**
- Modelos de mistura Gaussiana e algoritmo EM
- Variáveis latentes e análise fatorial
- Agrupamento espectral e constante de Cheeger
- Aprendizado de variedades (LLE, Isomap)

---

## 3. ALGORITMOS AVANÇADOS E OTIMIZAÇÃO

### 3.1 Métodos de Monte Carlo e Cadeias de Markov
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm

class MonteCarloMarkovChains:
    """
    Métodos de Monte Carlo e cadeias de Markov para ML
    """

    def __init__(self, target_distribution):
        self.target = target_distribution

    def metropolis_hastings_algorithm(self, initial_state, n_iterations=10000):
        """
        Algoritmo de Metropolis-Hastings para amostragem
        """
        class MetropolisHastings:
            def __init__(self, target_dist, initial_state):
                self.target = target_dist
                self.current_state = initial_state
                self.current_prob = self.target(initial_state)

                self.samples = [initial_state]
                self.accepted = 0

            def proposal_distribution(self, current_state):
                """Distribuição de proposta (Gaussiana simétrica)"""
                return np.random.normal(current_state, 0.5)

            def run_chain(self, n_iterations):
                """Executa cadeia de Markov"""
                for i in range(n_iterations):
                    # Propor nova amostra
                    proposed_state = self.proposal_distribution(self.current_state)
                    proposed_prob = self.target(proposed_state)

                    # Razão de aceitação
                    acceptance_ratio = proposed_prob / self.current_prob

                    # Aceitar ou rejeitar
                    if np.random.random() < acceptance_ratio:
                        self.current_state = proposed_state
                        self.current_prob = proposed_prob
                        self.accepted += 1

                    self.samples.append(self.current_state)

                acceptance_rate = self.accepted / n_iterations

                return {
                    'samples': np.array(self.samples),
                    'acceptance_rate': acceptance_rate,
                    'final_state': self.current_state
                }

        mh_sampler = MetropolisHastings(self.target, initial_state)
        results = mh_sampler.run_chain(n_iterations)

        return results

    def gibbs_sampling_algorithm(self, conditional_distributions, initial_state):
        """
        Amostragem de Gibbs para distribuições multivariadas
        """
        class GibbsSampler:
            def __init__(self, conditionals, initial):
                self.conditionals = conditionals  # Lista de distribuições condicionais
                self.n_variables = len(conditionals)
                self.current_state = initial.copy()

                self.samples = []

            def run_chain(self, n_iterations, burn_in=1000):
                """Executa amostragem de Gibbs"""
                for iteration in range(n_iterations + burn_in):
                    # Amostrar cada variável condicionalmente
                    for i in range(self.n_variables):
                        # p(x_i | x_{-i})
                        conditional = self.conditionals[i]
                        other_variables = [self.current_state[j] for j in range(self.n_variables) if j != i]

                        # Amostrar da condicional
                        self.current_state[i] = conditional(*other_variables)

                    # Armazenar amostra (após burn-in)
                    if iteration >= burn_in:
                        self.samples.append(self.current_state.copy())

                return {
                    'samples': np.array(self.samples),
                    'burn_in': burn_in,
                    'total_iterations': n_iterations + burn_in
                }

        gibbs_sampler = GibbsSampler(conditional_distributions, initial_state)
        results = gibbs_sampler.run_chain(n_iterations)

        return results

    def hamiltonian_monte_carlo(self, log_probability, initial_position, n_iterations=1000):
        """
        Monte Carlo Hamiltoniano (HMC) para amostragem eficiente
        """
        class HamiltonianMC:
            def __init__(self, log_prob, initial_pos):
                self.log_prob = log_prob
                self.position = initial_pos
                self.dimension = len(initial_pos)

                # Parâmetros HMC
                self.epsilon = 0.1  # Tamanho do passo
                self.L = 10  # Número de passos de Leapfrog

            def leapfrog_integration(self, position, momentum):
                """Integração Leapfrog para dinâmica Hamiltoniana"""
                # Meio passo para momentum
                gradient = self._numerical_gradient(self.log_prob, position)
                momentum_half = momentum - 0.5 * self.epsilon * gradient

                # Passos completos para posição e momentum
                for _ in range(self.L):
                    # Atualizar posição
                    position = position + self.epsilon * momentum_half

                    # Atualizar momentum (meio passo)
                    gradient = self._numerical_gradient(self.log_prob, position)
                    if _ != self.L - 1:  # Não no último passo
                        momentum_half = momentum_half - self.epsilon * gradient

                # Meio passo final para momentum
                gradient = self._numerical_gradient(self.log_prob, position)
                momentum = momentum_half - 0.5 * self.epsilon * gradient

                return position, momentum

            def hamiltonian_dynamics(self, n_iterations):
                """Dinâmica Hamiltoniana completa"""
                samples = []
                acceptance_count = 0

                current_position = self.position.copy()

                for iteration in range(n_iterations):
                    # Amostrar momentum do cinético (Gaussiano)
                    current_momentum = np.random.normal(0, 1, self.dimension)

                    # Calcular energia inicial
                    current_energy = self._hamiltonian(current_position, current_momentum)

                    # Integrar dinâmica
                    proposed_position, proposed_momentum = self.leapfrog_integration(
                        current_position, current_momentum
                    )

                    # Calcular energia proposta
                    proposed_energy = self._hamiltonian(proposed_position, proposed_momentum)

                    # Razão de aceitação
                    energy_difference = proposed_energy - current_energy
                    acceptance_probability = min(1, np.exp(-energy_difference))

                    # Aceitar ou rejeitar
                    if np.random.random() < acceptance_probability:
                        current_position = proposed_position
                        acceptance_count += 1

                    samples.append(current_position.copy())

                acceptance_rate = acceptance_count / n_iterations

                return {
                    'samples': np.array(samples),
                    'acceptance_rate': acceptance_rate,
                    'final_position': current_position
                }

            def _hamiltonian(self, position, momentum):
                """Função Hamiltoniana H = U + K"""
                # Energia potencial U = -log_prob
                potential_energy = -self.log_prob(position)

                # Energia cinética K = 0.5 * p^T p (para distribuição Gaussiana)
                kinetic_energy = 0.5 * np.sum(momentum ** 2)

                return potential_energy + kinetic_energy

            def _numerical_gradient(self, f, x, h=1e-5):
                """Gradiente numérico"""
                gradient = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h

                    gradient[i] = (f(x_plus) - f(x_minus)) / (2 * h)

                return gradient

        hmc_sampler = HamiltonianMC(log_probability, initial_position)
        results = hmc_sampler.hamiltonian_dynamics(n_iterations)

        return results

    def importance_sampling_estimation(self, target_distribution, proposal_distribution,
                                     n_samples=10000):
        """
        Amostragem por importância para estimação de expectativas
        """
        class ImportanceSampling:
            def __init__(self, target, proposal):
                self.target = target
                self.proposal = proposal

            def estimate_expectation(self, function, n_samples):
                """Estima E_p[f(X)] usando amostragem por importância"""
                samples = []
                weights = []

                for _ in range(n_samples):
                    # Amostrar da proposta
                    x = self.proposal.sample()
                    samples.append(x)

                    # Calcular peso de importância
                    target_prob = self.target(x)
                    proposal_prob = self.proposal(x)

                    if proposal_prob > 0:
                        weight = target_prob / proposal_prob
                    else:
                        weight = 0

                    weights.append(weight)

                # Estimativa ponderada
                samples = np.array(samples)
                weights = np.array(weights)

                # Normalizar pesos
                normalized_weights = weights / np.sum(weights)

                # Estimar expectativa
                expectation = np.sum(normalized_weights * np.array([function(x) for x in samples]))

                # Estimar variância
                variance = np.sum(normalized_weights * (np.array([function(x) for x in samples]) - expectation)**2)

                return {
                    'expectation': expectation,
                    'variance': variance,
                    'effective_sample_size': 1 / np.sum(normalized_weights**2),
                    'samples': samples,
                    'weights': weights
                }

        importance_sampler = ImportanceSampling(target_distribution, proposal_distribution)
        results = importance_sampler.estimate_expectation(lambda x: x**2, n_samples)  # Exemplo: E[X²]

        return results

    def sequential_monte_carlo(self, observations, initial_distribution, transition_model):
        """
        Monte Carlo Sequencial (SMC) para filtragem
        """
        class SequentialMC:
            def __init__(self, observations, initial_dist, transition):
                self.observations = observations
                self.initial_dist = initial_dist
                self.transition = transition
                self.n_particles = 1000

            def particle_filter(self):
                """Filtro de partículas"""
                n_timesteps = len(self.observations)

                # Inicializar partículas
                particles = self.initial_dist.sample(self.n_particles)
                weights = np.ones(self.n_particles) / self.n_particles

                particle_history = [particles.copy()]
                weight_history = [weights.copy()]

                for t in range(1, n_timesteps):
                    # Predição
                    particles = np.array([self.transition(particle) for particle in particles])

                    # Atualização de pesos baseada na observação
                    for i in range(self.n_particles):
                        likelihood = self._observation_likelihood(particles[i], self.observations[t])
                        weights[i] *= likelihood

                    # Normalizar pesos
                    weights_sum = np.sum(weights)
                    if weights_sum > 0:
                        weights /= weights_sum
                    else:
                        # Reamostragem degenerada
                        weights = np.ones(self.n_particles) / self.n_particles

                    # Reamostragem
                    if 1 / np.sum(weights**2) < self.n_particles / 2:
                        particles, weights = self._resample(particles, weights)

                    particle_history.append(particles.copy())
                    weight_history.append(weights.copy())

                return {
                    'particle_history': particle_history,
                    'weight_history': weight_history,
                    'final_particles': particles,
                    'final_weights': weights
                }

            def _observation_likelihood(self, particle, observation):
                """Verossimilhança da observação dado o estado"""
                # Implementação simplificada
                return np.exp(-0.5 * (particle - observation)**2)

            def _resample(self, particles, weights):
                """Reamostragem sistemática"""
                n_particles = len(particles)
                cumulative_weights = np.cumsum(weights)
                cumulative_weights /= cumulative_weights[-1]

                # Amostragem sistemática
                positions = (np.arange(n_particles) + np.random.random()) / n_particles

                new_particles = []
                new_weights = np.ones(n_particles) / n_particles

                j = 0
                for i in range(n_particles):
                    while j < n_particles - 1 and positions[i] > cumulative_weights[j]:
                        j += 1
                    new_particles.append(particles[j].copy())

                return np.array(new_particles), new_weights

        smc_filter = SequentialMC(observations, initial_distribution, transition_model)
        results = smc_filter.particle_filter()

        return results
```

**Métodos de Monte Carlo:**
- Algoritmo Metropolis-Hastings
- Amostragem de Gibbs
- Monte Carlo Hamiltoniano
- Amostragem por importância
- Monte Carlo sequencial

---

## 4. CONSIDERAÇÕES FINAIS

A teoria do aprendizado de máquina representa a base matemática e computacional para compreender os limites e possibilidades do aprendizado automático. Os conceitos apresentados fornecem o arcabouço teórico necessário para:

1. **Compreensão Profunda**: Fundamentos matemáticos sólidos do ML
2. **Análise de Limites**: Avaliação de capacidades e restrições
3. **Desenvolvimento de Algoritmos**: Base para criar novos métodos
4. **Avaliação de Performance**: Métricas teóricas de qualidade
5. **Pesquisa Avançada**: Ferramentas para investigação teórica

**Próximos Passos Recomendados**:
1. Dominar análise funcional e otimização convexa
2. Estudar teoria da informação e complexidade computacional
3. Explorar teoria do aprendizado estatístico
4. Desenvolver intuição para métodos de Monte Carlo
5. Contribuir para avanços teóricos em ML

---

*Documento preparado para fine-tuning de IA em Aprendizado de Máquina Teórico*
*Versão 1.0 - Preparado para implementação prática*
