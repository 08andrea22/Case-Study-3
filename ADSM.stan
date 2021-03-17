// Basic model: random intercepts from distribution with common mean, var
// No global intercept, all categories

data {
    int N; // number of observations
    int M; // number of groups (hospitals)
    int K; // number of predictors
    int P; // number of hospital covariates
    
    int y[N]; // "successes": deaths
    int n[N]; // trials: number of procedures
    row_vector[K] x[N]; // predictors
    matrix[M,P] z; // hospital level covariates (intercept NOT included)
    int g[N];    // map obs to groups (observations to hospitals)
}
parameters {
    real alpha; // global intercept
    real a[M]; // hospital random intercepts
    vector[P] xi; // hospital regression parameters
    vector[K] beta; // fixed effects coefficients
    real<lower=0> sigma2;  // IG(1,1) hyperprior for sigma2
}
transformed parameters{
  vector[M] mu; // random effects linear predictor
  real<lower=0> sigma; // standard deviation
  mu = z * xi;
  sigma = sqrt(sigma2);
}
model {
  alpha ~ normal(0,2.5); // global intercept
  a ~ normal(alpha + mu,sigma); // centered random intercepts
  xi ~ normal(0, sigma); // regression coefficients for RE mean
  beta ~ normal(0,2.5); //fixed effects
  for(i in 1:N) {
    y[i] ~ binomial(n[i], inv_logit( a[g[i]] + x[i]*beta));
  }
  sigma2 ~ inv_gamma(1,1); // exp(1) precision prior
}

