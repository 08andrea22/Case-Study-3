// Basic model: random intercepts from distribution with common mean, var
// No global intercept, all categories

data {
    int N; // number of observations
    int M; // number of groups (hospitals)
    int K; // number of predictors
    int P; // number of hospital covariates
    
    int y[N]; // "successes": deaths
    int n[N]; // trials: number of procedures
    row_vector[K] x[N]; // predictors (intercept included)
    matrix[M,P] z; // hospital level covariates (intercept included)
    int g[N];    // map obs to groups (observations to hospitals)
}
parameters {
    real alpha; // global intercept (inside of RE for centered param)
    vector[M] a; // hospital random intercepts 
    vector[P] xi; // hospital regression parameters
    vector[K] beta; // fixed effects coefficients
    real<lower=0> sigma2;  // random effect variance
    real<lower=0> tau2; // random effect reg coeff variance component
}
transformed parameters{
  vector[M] mu; // random effects linear predictor
  real<lower=0> sigma; // standard deviation
  real<lower=0> tau; // RE sd component
  vector[M] gamma;
  mu = z * xi;
  sigma = sqrt(sigma2);
  tau = sqrt(tau2);
  gamma = a - alpha;
}
model {
  a ~ normal(alpha + mu,sigma); // centered random intercepts
  alpha ~ normal(0, 2.5); // global intercept in the RE
  xi ~ normal(0, tau*sigma); // regression coefficients for RE mean
  beta ~ normal(0,2.5); //fixed effects
  for(i in 1:N) {
    y[i] ~ binomial(n[i], inv_logit( a[g[i]] + x[i]*beta));
  }
  sigma2 ~ inv_gamma(1,1); // exp(1) precision prior
  tau2 ~ inv_gamma(1,1); // exp(1) precision prior
}
generated quantities{
  // posterior predictive samples
  real y_new[N];
  for (i in 1:N) {
    y_new[i] = binomial_rng(n[i], inv_logit(a[g[i]] + x[i]*beta));
  }
}

