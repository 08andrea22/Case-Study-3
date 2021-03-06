// Basic model: random intercepts from distribution with common mean, var
// No global intercept, all categories

data {
    int N; // number of observations
    int M; // number of groups (hospitals)
    int K; // number of predictors
    int P; // number of hospital covariates for Z
    int Q; // number of hospital covariates for V
    
    int y[N]; // "successes": deaths
    int n[N]; // trials: number of procedures
    row_vector[K] x[N]; // predictors (intercept not included)
    matrix[M,P] z; // hospital level covariates for mu (intercept not included)
    matrix[M,Q] V; // hospital level covariates for v (intercept not included)
    int g[N];    // map obs to groups (observations to hospitals)
}
parameters {
    real alpha; // global intercept (inside of RE for centered param)
    vector[M] a; // hospital random intercepts 
    vector[P] xi; // hospital mean regression parameters
    vector[Q] delta; // hospital variance regression parameters
    vector[K] beta; // fixed effects coefficients
    real<lower=0> sigma2;  // random effect variance
    real<lower=0> tau2; // random effect reg coeff variance component
    real<lower=0> epsilon2; // random effect var reg variance
}
transformed parameters{
  vector[M] mu; // random effects linear predictor
  real<lower=0> sigma; // standard deviation
  real<lower=0> tau; // RE sd component
  real<lower=0> epsilon; // RE variance sd component
  vector[M] v; // log-linear variance regression 
  vector[M] gamma;
  mu = z * xi;
  sigma = sqrt(sigma2);
  tau = sqrt(tau2);
  epsilon = sqrt(epsilon2);
  v = sqrt(exp(V * delta)); // note: this is square rooted
  gamma = a - alpha;
}
model {
  a ~ normal(alpha + mu, sigma * v); // centered random intercepts
  alpha ~ normal(0, 2.5); // global intercept in the RE
  xi ~ normal(0, tau*sigma); // regression coefficients for RE mean
  delta ~ normal(0, epsilon * sigma); // regression coeffs for RE var
  beta ~ normal(0,2.5); //fixed effects
  for(i in 1:N) {
    y[i] ~ binomial(n[i], inv_logit( a[g[i]] + x[i]*beta));
  }
  sigma2 ~ inv_gamma(1,1); // exp(1) precision prior
  tau2 ~ inv_gamma(1,1); // exp(1) precision prior
  epsilon2 ~ inv_gamma(1,1); // exp(1) precision prior
}
generated quantities{
  // posterior predictive samples
  real y_new[N];
  for (i in 1:N) {
    y_new[i] = binomial_rng(n[i], inv_logit(a[g[i]] + x[i]*beta));
  }
}

