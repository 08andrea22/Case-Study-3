// Basic model: random intercepts from distribution with common mean, var
// No global intercept, all categories

data {
    int N; // number of observations
    int M; // number of groups (hospitals)
    int K; // number of predictors
    
    int y[N]; // "successes": deaths
    int n[N]; // trials: number of procedures
    row_vector[K] x[N]; // predictors
    int g[N];    // map obs to groups (observations to hospitals)
}
parameters {
    //real alpha; // global intercept
    real a[M]; // hospital random intercepts
    vector[K] beta; // fixed effects coefficients
    real<lower=0,upper=10> sigma;  // Unif(0,10) hyperprior for sigma
}
model {
  //alpha ~ normal(0,2.5); // global intercept
  a ~ normal(0,sigma); // random intercepts are independent
  beta ~ normal(0,2.5); //fixed effects
  for(i in 1:N) {
    y[i] ~ binomial(n[i], inv_logit( a[g[i]] + x[i]*beta));
  }
}

