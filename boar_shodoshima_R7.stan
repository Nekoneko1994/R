data {
  int<lower=2> T;                       // 年数（古→新）
  int<lower=0> culled[T];               // 年度別 総捕獲数
  vector<lower=0>[T] forest_area;       // 年度別 森林面積 [km^2]

  // --- 出没件数（観測がある年のみ渡す；無ければ M_sight=0 でOK） ---
  int<lower=0> M_sight;
  int<lower=1, upper=T> sight_idx[M_sight];
  vector<lower=0>[M_sight] sightings;

  // H28(2016)以降=1, 以前=0（捕獲体制シフト）
  int<lower=0, upper=1> cull_is_post[T];

  // 初期個体数の弱情報事前
  real log_N1_mean;
  real<lower=0> log_N1_sd;

  // --- アンカー（昨年度の推定“中央値”へゆるく寄せる；空でも可） ---
  int<lower=0> M_anchorN;
  int<lower=1, upper=T> anchorN_idx[M_anchorN];
  vector<lower=0>[M_anchorN] anchorN50;
  real<lower=0> sigma_anchorN;   // log スケールのノイズ（例 0.25）
}

transformed data {
  vector[T] cull_is_post_vec;
  for (t in 1:T) cull_is_post_vec[t] = cull_is_post[t];
}

parameters {
  // ---- 状態 ----
  vector<lower=1e-3>[T] N;             // 年度末 総個体数（頭）

  // 自然増加率 r_t を 0.9〜1.3 に制約：ロジット前パラメータのRW
  vector[T] theta_r;
  real mu_theta;
  real<lower=0> sigma_theta;

  // 捕獲率のロジット（RW）＋H28以降シフト
  vector[T] base_rw;
  real alpha_cull;
  real<lower=0> sigma_cull_rw;
  real delta_post;

  // 観測ノイズとリンク
  real log_sight_rate;                  // 出没と密度のリンク（対数）
  real<lower=0> sigma_proc;             // 状態ノイズ（log）
  real<lower=0> sigma_cull;             // 捕獲観測ノイズ（log）
  real<lower=0> sigma_sight;            // 出没観測ノイズ（log）
}

transformed parameters {
  vector<lower=0.9, upper=1.3>[T] r;
  for (t in 1:T) r[t] = 0.9 + 0.4 * inv_logit(theta_r[t]);

  vector[T] logit_cull_rate = base_rw + delta_post * cull_is_post_vec;
  vector<lower=0, upper=1>[T] cull_rate = inv_logit(logit_cull_rate);
}

model {
  // ---- 事前 ----
  target += normal_lpdf(log(N[1]) | log_N1_mean, log_N1_sd);

  mu_theta    ~ normal(0, 0.5);
  sigma_theta ~ normal(0, 0.3);
  theta_r[1]  ~ normal(mu_theta, 0.4);
  for (t in 2:T) theta_r[t] ~ normal(theta_r[t-1], sigma_theta);

  alpha_cull     ~ normal(0, 2);
  sigma_cull_rw  ~ normal(0, 0.5);
  base_rw[1]     ~ normal(alpha_cull, 1.0);
  for (t in 2:T) base_rw[t] ~ normal(base_rw[t-1], sigma_cull_rw);
  delta_post     ~ normal(0, 1);

  log_sight_rate ~ normal(0, 3);
  sigma_proc  ~ normal(0, 0.5);
  sigma_cull  ~ normal(0, 0.8);
  sigma_sight ~ normal(0, 1.0);

  // ---- 過程（前年 r*N から当年捕獲を引く）----
  for (t in 2:T) {
    real mean_post = fmax(1e-3, r[t] * N[t-1] - culled[t]);
    N[t] ~ lognormal(log(mean_post), sigma_proc);
  }

  // ---- 観測 ----
  // 捕獲：log(culled)
  for (t in 1:T) {
    target += normal_lpdf(log(culled[t] + 1e-9)
               | log(cull_rate[t] * (N[t] + culled[t])), sigma_cull);
  }
  // 出没（観測がある年だけループ；M_sight=0 なら回らない）
  for (m in 1:M_sight) {
    int tt = sight_idx[m];
    target += normal_lpdf(log(fmax(1e-9, sightings[m]))
               | (log_sight_rate + log((N[tt] + 0.5 * culled[tt]) / forest_area[tt])),
               sigma_sight);
  }

  // ---- アンカー（個体数の中央値；任意）----
  for (m in 1:M_anchorN) {
    target += normal_lpdf(log(N[anchorN_idx[m]]) | log(anchorN50[m]), sigma_anchorN);
  }
}

generated quantities {
  vector[T] abundance = N;
  vector[T] r_out = r;
}
