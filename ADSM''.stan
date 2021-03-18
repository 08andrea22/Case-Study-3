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
    real dist1[M,M];
    real dist2[M,M];
    real dist3[M,M];
}
parameters {
    vector[M] a; // hospital random intercepts 
    vector[P] xi; // hospital regression parameters
    vector[K] beta; // fixed effects coefficients
    real<lower=0> sigma2;  // random effect variance
    real<lower=0> phi1;
    real<lower=0> phi2;
    real<lower=0> phi3;
}
transformed parameters{
  vector[M] mu; // random effects linear predictor
  real<lower=0> sigma; // standard deviation
  cov_matrix[M] SIGMA;
  mu = z * xi;
  sigma = sqrt(sigma2);
  for(j in 1:M){
    for(jj in 1:M){
      SIGMA[j,jj]=sigma2*exp(-phi1*dist1[j,jj]-phi2*dist2[j,jj]-phi3*dist3[j,jj]);
    }
  }
}
model {
  a ~ multi_normal(rep_vector(0,M), SIGMA);//normal(mu,sigma); // centered random intercepts
  xi ~ normal(0, sigma); // regression coefficients for RE mean
  beta ~ normal(0,2.5); //fixed effects
  for(i in 1:N) {
    y[i] ~ binomial(n[i], inv_logit( a[g[i]] + x[i]*beta));
  }
  sigma2 ~ inv_gamma(1,1); // exp(1) precision prior
  phi1 ~ gamma(10,1);
  phi2 ~ gamma(10,1);
  phi3 ~ gamma(10,1);
}





