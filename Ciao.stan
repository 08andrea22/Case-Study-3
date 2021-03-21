
// Basic model: random intercepts and random intecept variance
// RE covariance: dependent
// Global intercept is within RE, Category 1 dropped
data {
    int N; // number of training observations
    int M; // number of groups (hospitals)
    int K; // number of predictors
    int P; // number of hospital covariates for Z
    int Q; // number of hospital covariates for V
    int r; // number of clusters
    
    int y[N]; // "successes": deaths
    int n[N]; // trials: number of procedures
    row_vector[K] x[N]; // predictors (intercept not included)
    matrix[M,P] z; // hospital level covariates for mu (intercept not included)
    matrix[M,Q] V; // hospital level covariates for v (intercept not included)
    matrix[N, 4 * r] q; // matrix of interactions
    int g[N];    // map obs to groups (observations to hospitals)
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
    vector[4 * r] kappa; // cluster-category interactions
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
  vector[N] G_rep; // vector needed to calculate z's
  vector[N] Z1; //a
  vector[N] Z2;
  vector[N] Z3;
  vector[N] Z4;
  vector[N] Z5;
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
  G_rep = gamma[g];
  Z1 = G_rep;
  Z2 = G_rep + q[, 1 : r] * kappa[1:r] ;
  Z3 = G_rep + q[, (r+1):2*r] * kappa[(r+1):2*r];
  Z4 = G_rep + q[, (2*r + 1):3*r] * kappa[(2*r + 1):3*r];
  Z5 = G_rep + q[, (3*r + 1): 4*r] * kappa[(3*r + 1): 4*r];
}
model {
  a ~ multi_normal(alpha + mu, SIGMA);// centered dependent random intercepts
  alpha ~ normal(0, 2.5); // global intercept in the RE
  xi ~ normal(0, tau*sigma); // regression coefficients for RE mean
  delta ~ normal(0, epsilon * sigma); // regression coeffs for RE var
  beta ~ normal(0, 2.5); //fixed effects
  for(i in 1:N) {
    y[i] ~ binomial(n[i], inv_logit( a[g[i]] + x[i]*beta +  q[i] * kappa));
  }
  sigma2 ~ inv_gamma(1,1); // exp(1) precision prior
  tau2 ~ inv_gamma(1,1); // exp(1) precision prior
  epsilon2 ~ inv_gamma(1,1); // exp(1) precision prior
  phi ~ gamma(1,1);
  //phi2 ~ gamma(1,1);
  //phi3 ~ gamma(0.01,0.01);
}

