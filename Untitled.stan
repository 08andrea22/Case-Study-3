
// Basic model: random intercepts and random intecept variance
// RE covariance: dependent
// Global intercept is within RE, Category 1 dropped

data {
    int N; // number of training observations
    int N_test; // number of test observations
    int M; // number of groups (hospitals)
    int K; // number of predictors
    int P; // number of hospital covariates for Z
    int Q; // number of hospital covariates for V
    
    int y[N]; // "successes": deaths
    int n[N]; // trials: number of procedures
    int n_test[N_test]; // trials for testing data
    row_vector[K] x[N]; // predictors (intercept not included)
    row_vector[K] x_test[N_test]; // predictors for test data
    matrix[M,P] z; // hospital level covariates for mu (intercept not included)
    matrix[M,Q] V; // hospital level covariates for v (intercept not included)
    int g[N];    // map obs to groups (observations to hospitals)
    int g_test[N_test]; // map testing obs to groups
    real dist[M,M]; //dist 1 for corr matrix
    //real dist2[M,M]; //dist 2 for corr matrix
    //real dist3[M,M]; //pd correction factor for corr matrix
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
    real<lower=0> phi; // RE corr parameter
    //real<lower=0> phi2; // RE corr parameter
    //real<lower=0> phi3; // RE corr parameter
}
transformed parameters{
  vector[M] mu; // random effects linear predictor
  real<lower=0> sigma; // standard deviation
  real<lower=0> tau; // RE sd component
  real<lower=0> epsilon; // RE variance sd component
  vector[M] v; // log-linear variance regression 
  vector[M] gamma; // de-centered RE intercepts
  matrix[M, M] R; // RE correlation matrix 
  cov_matrix[M] SIGMA; // RE cov matrix
  mu = z * xi;
  sigma = sqrt(sigma2);
  tau = sqrt(tau2);
  epsilon = sqrt(epsilon2);
  v = sqrt(exp(V * delta)); // note: this is square rooted
  gamma = a - alpha;
  for(j in 1:M){
    for(jj in 1:M){
      R[j,jj]=exp(-phi*dist[j,jj]);
      //exp(-phi1*dist1[j,jj]-phi2*dist2[j,jj]-phi3*dist3[j,jj]);
    }
  }
  SIGMA = diag_matrix(v)' * R * diag_matrix(v);
}
model {
  a ~ multi_normal(alpha + mu, SIGMA);// centered dependent random intercepts
  alpha ~ normal(0, 2.5); // global intercept in the RE
  xi ~ normal(0, tau*sigma); // regression coefficients for RE mean
  delta ~ normal(0, epsilon * sigma); // regression coeffs for RE var
  beta ~ normal(0, 2.5); //fixed effects
  for(i in 1:N) {
    y[i] ~ binomial(n[i], inv_logit( a[g[i]] + x[i]*beta));
  }
  sigma2 ~ inv_gamma(1,1); // exp(1) precision prior
  tau2 ~ inv_gamma(1,1); // exp(1) precision prior
  epsilon2 ~ inv_gamma(1,1); // exp(1) precision prior
  phi ~ gamma(1,1);
  //phi2 ~ gamma(1,1);
  //phi3 ~ gamma(0.01,0.01);
}
generated quantities{
  // posterior predictive samples
  real y_new[N];
  real Prob[N];
  real y_test[N_test];
  for (i in 1:N) {
    y_new[i] = binomial_rng(n[i], inv_logit(a[g[i]] + x[i]*beta));
    Prob[i]=inv_logit( a[g[i]] + x[i]*beta);
  }
    
  for (i in 1:N_test) {
    y_test[i] = binomial_rng(n_test[i], inv_logit(a[g_test[i]] + x_test[i]*beta));
  }
}