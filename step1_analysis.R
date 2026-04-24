#stock direction prediction: AAPL, NFLX, MSFT, AMZN
#models: logistic regression, random forest, xgboost
#features: technical indicators + pca + k-means cluster label

library(tidyverse)
library(quantmod)
library(caret)
library(randomForest)
library(xgboost)
library(pROC)
library(TTR)
library(lubridate)
library(ggplot2)
library(zoo)
library(factoextra)

set.seed(123)

start <- as.Date("2015-01-01")
end   <- as.Date("2026-04-01")

stocks <- c("AAPL", "NFLX", "MSFT", "AMZN")

#download all four tickers
raw_list <- list()
for (ticker in stocks) {
  raw <- getSymbols(ticker, src = "yahoo",
                    from = start, to = end,
                    auto.assign = FALSE)
  raw_list[[ticker]] <- raw
}


#build technical indicator features for one xts object
get_indicators <- function(stock_xts, period = 1) {
  ad  <- Ad(stock_xts)
  hi  <- Hi(stock_xts)
  lo  <- Lo(stock_xts)
  vo  <- Vo(stock_xts)
  M <-data.frame(h = as.numeric(hi),
                 l = as.numeric(lo),
                 c = as.numeric(ad))
  rownames(M)<-index(lo)
  hlac <- as.xts(M)
  
  #target: price goes up (1) or down (0) after `period` days
  price_change <- as.numeric(ad) - as.numeric(Ad(lag(stock_xts, period)))
  response     <- ifelse(price_change > 0, "UP", "DOWN")
  
  rsi  <- as.numeric(RSI(ad, n = 14))
  sto  <- stoch(hlac, nFastK = 14) * 100
  wpr  <- as.numeric(WPR(hlac, n = 14)) * (-100)
  macd_obj <- MACD(ad, nFast = 12, nSlow = 26, nSig = 9)
  roc_ind  <- as.numeric(ROC(ad, n = 14)) * 100
  obv_ind  <- as.numeric(OBV(ad, vo))
  
  #rolling mean ratios (weekly, quarterly, annual)
  close_num <- as.numeric(ad)
  weekly_mean    <- rollmean(close_num, k = 7,   fill = NA, align = "right")
  quarterly_mean <- rollmean(close_num, k = 90,  fill = NA, align = "right")
  annual_mean    <- rollmean(close_num, k = 252, fill = NA, align = "right")
  
  weekly_ratio    <- weekly_mean    / close_num
  quarterly_ratio <- quarterly_mean / close_num
  annual_ratio    <- annual_mean    / close_num
  ann_wk_ratio    <- annual_mean    / weekly_mean
  ann_qt_ratio    <- annual_mean    / quarterly_mean
  
  #intraday ratios
  open_num <- as.numeric(Op(stock_xts))
  high_num <- as.numeric(hi)
  low_num  <- as.numeric(lo)
  open_close_ratio <- open_num / close_num
  high_close_ratio <- high_num / close_num
  low_close_ratio  <- low_num  / close_num
  
  df <- data.frame(
    RSI              = rsi,
    StoFastK         = as.numeric(sto[, 1]),
    StoFastD         = as.numeric(sto[, 2]),
    StoSlowD         = as.numeric(sto[, 3]),
    WilliamPR        = wpr,
    MACD             = as.numeric(macd_obj[, 1]),
    MACDSignal       = as.numeric(macd_obj[, 2]),
    PriceROC         = roc_ind,
    OBV              = obv_ind,
    WeeklyMeanRatio  = weekly_ratio,
    QuarterlyRatio   = quarterly_ratio,
    AnnualRatio      = annual_ratio,
    AnnWeeklyRatio   = ann_wk_ratio,
    AnnQuarterlyRatio = ann_qt_ratio,
    OpenCloseRatio   = open_close_ratio,
    HighCloseRatio   = high_close_ratio,
    LowCloseRatio    = low_close_ratio,
    Response         = response
  )
  
  #remove NA rows from indicator warm-up + lag
  df <- df[36:nrow(df), ]
  df <- head(df, nrow(df) - period)
  df$Response <- factor(df$Response, levels = c("DOWN", "UP"))
  df <- df[complete.cases(df), ]
  return(df)
}

feature_cols <- c("RSI","StoFastK","StoFastD","StoSlowD","WilliamPR",
                  "MACD","MACDSignal","PriceROC","OBV",
                  "WeeklyMeanRatio","QuarterlyRatio","AnnualRatio",
                  "AnnWeeklyRatio","AnnQuarterlyRatio",
                  "OpenCloseRatio","HighCloseRatio","LowCloseRatio")

#pca + k-means on one stock's feature matrix
run_pca_kmeans <- function(df, k = 3) {
  x_scaled <- scale(df[, feature_cols])
  pca_res  <- prcomp(x_scaled, center = FALSE, scale. = FALSE)
  scores   <- as.data.frame(pca_res$x[, 1:5])
  km       <- kmeans(scores, centers = k, nstart = 30)
  list(pca = pca_res, km = km, scores = scores,
       cluster = factor(km$cluster))
}

#train all three models for one ticker/period dataset
train_models <- function(df) {
  # 80/20 time-based split first, before any unsupervised learning
  n         <- nrow(df)
  split_idx <- floor(n * 0.8)
  train     <- df[1:split_idx, ]
  test      <- df[(split_idx + 1):n, ]
  
  # fit pca only on training data
  train_scaled <- scale(train[, feature_cols])
  pca_center   <- attr(train_scaled, "scaled:center")
  pca_scale    <- attr(train_scaled, "scaled:scale")
  pca_fit      <- prcomp(train_scaled, center = FALSE, scale. = FALSE)
  
  # apply the same pca transform to test data using training mean/sd
  test_scaled  <- scale(test[, feature_cols],
                        center = pca_center,
                        scale  = pca_scale)
  train_scores <- as.data.frame(pca_fit$x[, 1:5])
  test_scores  <- as.data.frame(
    predict(pca_fit, test_scaled)[, 1:5]
  )
  #fit kmeans only on training pca scores
  km_fit       <- kmeans(train_scores, centers = 3, nstart = 30)
  
  # assign cluster labels using training centroids
  #for test: find nearest centroid by euclidean distance
  assign_cluster <- function(scores_df, centers) {
    apply(scores_df, 1, function(row) {
      dists <- apply(centers, 1, function(c) sqrt(sum((row - c)^2)))
      which.min(dists)
    })
  }
  
  cluster_levels <- as.character(1:3)
  
  train$cluster <- factor(km_fit$cluster, levels = cluster_levels)
  test$cluster  <- factor(
    assign_cluster(test_scores, km_fit$centers),
    levels = cluster_levels
  )
  all_feats <- c(feature_cols, "cluster")
  
  X_tr <- train[, all_feats]
  X_te <- test[,  all_feats]
  y_tr <- train$Response
  y_te <- test$Response
  

  
  ctrl <- trainControl(method = "cv", number = 5,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary,
                       savePredictions = "final")
  
  #logistic regression
  lr <- train(x = X_tr, y = y_tr,
              method = "glm", family = "binomial",
              metric = "ROC", trControl = ctrl,
              preProcess = c("center","scale"))
  
  #random forest
  rf <- train(x = X_tr, y = y_tr,
              method = "rf", metric = "ROC",
              trControl = ctrl,
              tuneGrid = expand.grid(mtry = c(3, 5)),
              ntree = 300)
  
  #xgboost
  train_data <- cbind(y = y_tr, X_tr)
  
  xgb <- train(y ~ ., data = train_data,
               method = "xgbTree", metric = "ROC",
               trControl = ctrl, verbosity = 0,
               tuneGrid = expand.grid(
                 nrounds = c(100, 200), max_depth = c(3, 5),
                 eta = c(0.05, 0.1), gamma = 0,
                 colsample_bytree = 0.8,
                 min_child_weight = 1, subsample = 0.8))
  pca_km_result <- list(
    pca        = pca_fit,
    km         = km_fit,
    scores     = train_scores,
    cluster    = train$cluster,
    center     = pca_center,
    scale      = pca_scale
  )
  
  list(lr = lr, rf = rf, xgb = xgb,
       train = train, test = test,
       X_tr = X_tr, X_te = X_te,
       y_tr = y_tr, y_te = y_te,
       pca_km = pca_km_result,
       df = df)
}

#extract evaluation metrics
get_metrics <- function(model, X_te, y_te, model_name) {
  pred <- predict(model, X_te)
  prob <- predict(model, X_te, type = "prob")[, "UP"]
  cm   <- confusionMatrix(pred, y_te, positive = "UP")
  roc_obj <- roc(as.integer(y_te == "UP"), prob, quiet = TRUE)
  data.frame(
    model       = model_name,
    accuracy    = round(cm$overall["Accuracy"],    4),
    sensitivity = round(cm$byClass["Sensitivity"], 4),
    specificity = round(cm$byClass["Specificity"], 4),
    precision   = round(cm$byClass["Precision"],   4),
    recall      = round(cm$byClass["Recall"],      4),
    f1          = round(cm$byClass["F1"],           4),
    auc         = round(auc(roc_obj),               4),
    row.names   = NULL
  )
}

#confusion matrix ggplot helper
plot_cm <- function(cm_obj, title_str) {
  tbl <- as.data.frame(cm_obj$table)
  colnames(tbl) <- c("predicted","actual","n")
  ggplot(tbl, aes(x = actual, y = predicted, fill = n)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = n), size = 8, fontface = "bold", color = "white") +
    scale_fill_gradient(low = "#90CAF9", high = "#1565C0") +
    labs(title = title_str, x = "actual", y = "predicted") +
    theme_bw(base_size = 13) +
    theme(legend.position = "none", panel.grid = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold"))
}

#store all results
all_results  <- list()
all_metrics  <- list()
all_roc_data <- list()
all_cm_data  <- list()
all_imp_rf   <- list()
all_imp_xgb  <- list()
all_coef_lr  <- list()
all_pca_km   <- list()

for (ticker in stocks) {
  cat("processing", ticker, "...\n")
  df  <- get_indicators(raw_list[[ticker]], period = 1)
  res <- train_models(df)
  
  X_te <- res$X_te
  y_te <- res$y_te
  
  m_lr  <- get_metrics(res$lr,  X_te, y_te, "Logistic Regression")
  m_rf  <- get_metrics(res$rf,  X_te, y_te, "Random Forest")
  m_xgb <- get_metrics(res$xgb, X_te, y_te, "XGBoost")
  all_metrics[[ticker]] <- bind_rows(m_lr, m_rf, m_xgb)
  
  #roc data for all three models
  get_roc_df <- function(model, label) {
    prob    <- predict(model, X_te, type = "prob")[, "UP"]
    roc_obj <- roc(as.integer(y_te == "UP"), prob, quiet = TRUE)
    data.frame(fpr   = 1 - roc_obj$specificities,
               tpr   = roc_obj$sensitivities,
               model = paste0(label, "  AUC=",
                              round(auc(roc_obj), 3)))
  }
  all_roc_data[[ticker]] <- bind_rows(
    get_roc_df(res$lr,  "LR"),
    get_roc_df(res$rf,  "RF"),
    get_roc_df(res$xgb, "XGB")
  )
  
  #confusion matrix tables
  get_cm_tbl <- function(model) {
    pred <- predict(model, X_te)
    cm   <- confusionMatrix(pred, y_te, positive = "UP")
    as.data.frame(cm$table)
  }
  all_cm_data[[ticker]] <- list(
    lr  = get_cm_tbl(res$lr),
    rf  = get_cm_tbl(res$rf),
    xgb = get_cm_tbl(res$xgb)
  )
  
  #feature importance
  all_imp_rf[[ticker]] <- varImp(res$rf)$importance %>%
    rownames_to_column("feature") %>% arrange(desc(Overall)) %>% head(17)
  all_imp_xgb[[ticker]] <- varImp(res$xgb)$importance %>%
    rownames_to_column("feature") %>% arrange(desc(Overall)) %>% head(17)
  
  #logistic regression coefficients
  coef_raw <- coef(res$lr$finalModel)
  se <- summary(res$lr$finalModel)$coefficients[, "Std. Error"]
  all_coef_lr[[ticker]] <- data.frame(
    feature   = names(coef_raw),
    coef      = as.numeric(coef_raw)
  ) %>%
    filter(feature != "(Intercept)") %>%
    mutate(OR       = round(exp(coef), 4),
           OR_lower = round(exp(coef - 1.96 * se), 4),
           OR_upper = round(exp(coef + 1.96 * se), 4),
           direction = ifelse(coef > 0, "positive", "negative")) %>%
    arrange(desc(abs(coef)))
  
  #pca + kmeans results
  all_pca_km[[ticker]] <- res$pca_km
  all_results[[ticker]] <- res
}

#print summaries
for (ticker in stocks) {
  cat("\n---", ticker, "---\n")
  print(all_metrics[[ticker]])
}

#plot examples for the first ticker (AAPL)
t <- "AAPL"

#roc curves
p_roc <- ggplot(all_roc_data[[t]], aes(x = fpr, y = tpr, color = model)) +
  geom_line(linewidth = 1.1) +
  geom_abline(linetype = "dashed", color = "gray60") +
  scale_color_manual(values = c("#1976D2","#43A047","#E53935")) +
  labs(title = paste("roc curves -", t),
       x = "false positive rate", y = "true positive rate", color = "") +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")
print(p_roc)

#confusion matrices (AAPL as example)
for (mod_name in c("lr","rf","xgb")) {
  cm_tbl <- all_cm_data[[t]][[mod_name]]
  colnames(cm_tbl) <- c("predicted","actual","n")
  p <- ggplot(cm_tbl, aes(x = actual, y = predicted, fill = n)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = n), size = 8, fontface = "bold", color = "white") +
    scale_fill_gradient(low = "#90CAF9", high = "#1565C0") +
    labs(title = paste("confusion matrix -", t, "-", toupper(mod_name)),
         x = "actual", y = "predicted") +
    theme_bw(base_size = 12) +
    theme(legend.position = "none", panel.grid = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold"))
  print(p)
}

#rf feature importance
p_rf_imp <- ggplot(all_imp_rf[[t]],
                   aes(x = reorder(feature, Overall), y = Overall)) +
  geom_col(fill = "#43A047", width = 0.65) +
  coord_flip() +
  labs(title = paste("random forest importance -", t), x = "", y = "gini") +
  theme_bw(base_size = 12)
print(p_rf_imp)

#xgb feature importance
p_xgb_imp <- ggplot(all_imp_xgb[[t]],
                    aes(x = reorder(feature, Overall), y = Overall)) +
  geom_col(fill = "#FB8C00", width = 0.65) +
  coord_flip() +
  labs(title = paste("xgboost importance -", t), x = "", y = "relative") +
  theme_bw(base_size = 12)
print(p_xgb_imp)

#lr forest plot
p_lr_forest <- ggplot(na.omit(all_coef_lr[[t]]),
                      aes(x = OR, y = reorder(feature, OR),
                          color = direction)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_errorbarh(aes(xmin = OR_lower, xmax = OR_upper),
                 height = 0.35, linewidth = 0.8) +
  geom_point(size = 3.5) +
  scale_color_manual(values = c("positive" = "#1976D2",
                                "negative" = "#E53935")) +
  labs(title = paste("lr forest plot -", t),
       x = "odds ratio", y = "", color = "") +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")+xlim(0,5)
print(p_lr_forest)

#pca scree plot
pca_obj <- all_pca_km[[t]]$pca
pca_imp <- summary(pca_obj)$importance
scree_df <- data.frame(
  PC     = 1:10,
  VarExp = pca_imp[2, 1:10] * 100,
  CumVar = pca_imp[3, 1:10] * 100
)

p_scree <- ggplot(scree_df, aes(x = PC, y = VarExp)) +
  geom_col(fill = "#1976D2", width = 0.65) +
  geom_line(aes(y = CumVar), color = "#E53935",
            linewidth = 1, linetype = "dashed") +
  geom_point(aes(y = CumVar), color = "#E53935", size = 2.5) +
  labs(title = paste("pca scree -", t),
       x = "principal component", y = "variance explained (%)") +
  theme_bw(base_size = 12)
print(p_scree)

#pca biplot (PC1 vs PC2 colored by cluster)
scores_df <- all_pca_km[[t]]$scores
scores_df$cluster <- all_pca_km[[t]]$cluster

p_cluster <- ggplot(scores_df, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.5, size = 1.5) +
  scale_color_manual(values = c("#1976D2","#43A047","#E53935")) +
  labs(title = paste("pca clusters (k=3) -", t),
       x = "PC1", y = "PC2", color = "cluster") +
  theme_bw(base_size = 12)
print(p_cluster)

#accuracy across 4 tickers bar chart
acc_compare <- map_dfr(stocks, function(tk) {
  all_metrics[[tk]] %>% mutate(ticker = tk)
})

p_acc_compare <- ggplot(acc_compare,
                        aes(x = ticker, y = accuracy, fill = model)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("#1976D2","#43A047","#E53935")) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "accuracy comparison across tickers and models",
       x = "ticker", y = "accuracy", fill = "") +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")
print(p_acc_compare)

#save all objects for shiny
saveRDS(all_metrics,  "all_metrics.rds")
saveRDS(all_roc_data, "all_roc_data.rds")
saveRDS(all_cm_data,  "all_cm_data.rds")
saveRDS(all_imp_rf,   "all_imp_rf.rds")
saveRDS(all_imp_xgb,  "all_imp_xgb.rds")
saveRDS(all_coef_lr,  "all_coef_lr.rds")
saveRDS(all_pca_km,   "all_pca_km.rds")
saveRDS(all_results,  "all_results.rds")
saveRDS(stocks,       "stocks.rds")
