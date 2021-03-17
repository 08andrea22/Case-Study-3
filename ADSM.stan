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
    vector[M] a; // hospital random intercepts 
    vector[P] xi; // hospital regression parameters
    vector[K] beta; // fixed effects coefficients
    real<lower=0> sigma2;  // random effect variance
}
transformed parameters{
  vector[M] mu; // random effects linear predictor
  real<lower=0> sigma; // standard deviation
  mu = z * xi;
  sigma = sqrt(sigma2);
}
model {
  a ~ normal(mu,sigma); // centered random intercepts
  xi ~ normal(0, sigma); // regression coefficients for RE mean
  beta ~ normal(0,2.5); //fixed effects
  for(i in 1:N) {
    y[i] ~ binomial(n[i], inv_logit( a[g[i]] + x[i]*beta));
  }
  sigma2 ~ inv_gamma(1,1); // exp(1) precision prior
}

