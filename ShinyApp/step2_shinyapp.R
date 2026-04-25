#package download first
library(shiny)
library(tidyverse)
library(quantmod)
library(caret)
library(randomForest)
library(xgboost)
library(pROC)
library(zoo)
library(lubridate)
library(DT)

safe_load <- function(path, default = NULL) {
  if (file.exists(path)) readRDS(path) else default
}

all_metrics  <- safe_load("all_metrics.rds",  list())
all_roc_data <- safe_load("all_roc_data.rds", list())
all_cm_data  <- safe_load("all_cm_data.rds",  list())
all_imp_rf   <- safe_load("all_imp_rf.rds",   list())
all_imp_xgb  <- safe_load("all_imp_xgb.rds",  list())
all_coef_lr  <- safe_load("all_coef_lr.rds",  list())
all_pca_km   <- safe_load("all_pca_km.rds",   list())
all_results  <- safe_load("all_results.rds",  list())
stocks       <- safe_load("stocks.rds", c("AAPL","NFLX","MSFT","AMZN"))

#confusion matrix tile helper
draw_cm <- function(cm_tbl, title_str) {
  colnames(cm_tbl) <- c("predicted","actual","n")
  ggplot(cm_tbl, aes(x = actual, y = predicted, fill = n)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = n), size = 8, fontface = "bold", color = "white") +
    scale_fill_gradient(low = "#90CAF9", high = "#1565C0") +
    labs(title = title_str, x = "actual", y = "predicted") +
    theme_bw(base_size = 13) +
    theme(legend.position = "none", panel.grid = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold"))
}

css <- "
  body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; }
  .navbar { background: #1565C0 !important; border: none;
             box-shadow: 0 2px 8px rgba(0,0,0,.2); }
  .navbar-brand, .navbar-nav > li > a { color: #fff !important; font-weight: 500; }
  .navbar-nav > li > a:hover { background: rgba(255,255,255,.15) !important; }
  .card { background: #fff; border-radius: 12px;
           box-shadow: 0 2px 12px rgba(0,0,0,.07);
           padding: 22px 24px; margin-bottom: 20px; }
  .section-title { font-size: .95em; font-weight: 700; color: #1565C0;
                    text-transform: uppercase; letter-spacing: .5px;
                    margin-bottom: 14px; }
  .metric-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
  .mbox { flex: 1; min-width: 100px; background: #fff; border-radius: 10px;
           box-shadow: 0 2px 8px rgba(0,0,0,.07); padding: 14px 8px;
           text-align: center; }
  .mbox-label { font-size: .72em; color: #9e9e9e; text-transform: uppercase;
                 letter-spacing: .4px; margin-bottom: 4px; }
  .mbox-value { font-size: 1.8em; font-weight: 700; color: #1565C0; line-height: 1; }
  select, .form-control { border-radius: 8px !important; border-color: #ddd !important; }
  .btn-primary { background: #1565C0; border-color: #1565C0; border-radius: 8px;
                  font-weight: 500; }
  .btn-primary:hover { background: #1976D2; border-color: #1976D2; }
  hr { border-color: #f0f0f0; margin: 14px 0; }
"

ui <- navbarPage(
  title = "Stock Direction Predictor",
  tags$head(tags$style(HTML(css))),
  
  #tab 1: overview - accuracy across all tickers
  tabPanel("Overview",
           fluidPage(br(),
                     fluidRow(
                       column(12,
                              div(class = "card",
                                  div(class = "section-title", "accuracy comparison across stocks and models"),
                                  selectInput("ov_metric", "metric:",
                                              choices = c("accuracy","sensitivity","specificity",
                                                          "precision","recall","f1","auc"),
                                              selected = "accuracy", width = "220px"),
                                  plotOutput("ov_bar", height = "320px")
                              )
                       )
                     ),
                     fluidRow(
                       column(12,
                              div(class = "card",
                                  div(class = "section-title", "all metrics table"),
                                  DTOutput("ov_tbl")
                              )
                       )
                     )
           )
  ),
  
  #tab 2: per-stock deep dive
  tabPanel("Model Results",
           fluidPage(br(),
                     fluidRow(
                       column(3,
                              div(class = "card",
                                  div(class = "section-title", "controls"),
                                  selectInput("sel_ticker", "ticker:",
                                              choices = stocks, selected = "AAPL")
                              )
                       )
                     ),
                     fluidRow(
                       column(7,
                              div(class = "card",
                                  div(class = "section-title", "metrics for selected ticker"),
                                  DTOutput("stock_metrics_tbl")
                              )
                       ),
                       column(5,
                              div(class = "card",
                                  div(class = "section-title", "roc curves - all three models"),
                                  plotOutput("roc_plot", height = "340px")
                              )
                       )
                     ),
                     fluidRow(
                       column(4,
                              div(class = "card",
                                  plotOutput("cm_lr_plot", height = "270px"))
                       ),
                       column(4,
                              div(class = "card",
                                  plotOutput("cm_rf_plot", height = "270px"))
                       ),
                       column(4,
                              div(class = "card",
                                  plotOutput("cm_xgb_plot", height = "270px"))
                       )
                     )
           )
  ),
  
  #tab 3: feature importance + lr coefficients
  tabPanel("Features",
           fluidPage(br(),
                     fluidRow(
                       column(3,
                              div(class = "card",
                                  div(class = "section-title", "controls"),
                                  selectInput("feat_ticker", "ticker:",
                                              choices = stocks, selected = "AAPL"),
                                  sliderInput("n_imp", "features shown:", 5, 17, 14, 1)
                              )
                       )
                     ),
                     fluidRow(
                       
                       column(6,
                              div(class = "card",
                                  div(class = "section-title", "random forest - feature importance"),
                                  plotOutput("rf_imp_plot", height = "360px")
                              )
                       ),
                       column(6,
                              div(class = "card",
                                  div(class = "section-title", "xgboost - feature importance"),
                                  plotOutput("xgb_imp_plot", height = "360px")
                              )
                       )
                     ),
                     fluidRow(
                       column(12,
                              div(class = "card",
                                  div(class = "section-title",
                                      "logistic regression - forest plot (odds ratios)"),
                                  plotOutput("lr_forest_plot", height = "400px")
                              )
                       )
                     ),
                     fluidRow(
                       column(12,
                              div(class = "card",
                                  div(class = "section-title",
                                      "logistic regression - coefficient table (HR = odds ratio)"),
                                  DTOutput("lr_coef_tbl")
                              )
                       )
                     )
           )
  ),
  
  #tab 4: pca + clustering
  tabPanel("PCA & Clustering",
           fluidPage(br(),
                     fluidRow(
                       column(3,
                              div(class = "card",
                                  div(class = "section-title", "controls"),
                                  selectInput("pca_ticker", "ticker:",
                                              choices = stocks, selected = "AAPL"),
                                  hr(),
                                  p(style = "font-size:.82em; color:#aaa; line-height:1.5;",
                                    "PCA applied to scaled technical features.
               K-means (k=3) run on top 5 PCs to detect market regimes.
               Cluster label is added as a feature in all supervised models.")
                              )
                       ),
                       column(9,
                              fluidRow(
                                column(6,
                                       div(class = "card",
                                           div(class = "section-title", "pca scree plot"),
                                           plotOutput("pca_scree", height = "280px"))
                                ),
                                column(6,
                                       div(class = "card",
                                           div(class = "section-title", "cumulative variance"),
                                           plotOutput("pca_cumvar", height = "280px"))
                                )
                              )
                       )
                     ),
                     fluidRow(
                       column(6,
                              div(class = "card",
                                  div(class = "section-title", "pc1 vs pc2 - colored by cluster (k=3)"),
                                  plotOutput("cluster_scatter", height = "340px"))
                       ),
                       column(6,
                              div(class = "card",
                                  div(class = "section-title", "cluster distribution"),
                                  plotOutput("cluster_bar", height = "340px"))
                       )
                     ),
                     fluidRow(
                       column(6,
                              div(class = "card",
                                  div(class = "section-title", "pc1 vs pc3"),
                                  plotOutput("cluster_pc13", height = "300px"))
                       ),
                       column(6,
                              div(class = "card",
                                  div(class = "section-title", "cluster vs direction (response)"),
                                  plotOutput("cluster_response", height = "300px"))
                       )
                     )
           )
  ),
  
  #tab 5: about
  tabPanel("About",
           fluidPage(br(),
                     fluidRow(
                       column(8, offset = 2,
                              div(class = "card",
                                  h4("research question"),
                                  p("can we predict whether AAPL, NFLX, MSFT, or AMZN will go UP
               or DOWN the next trading day using technical indicators?"),
                                  hr(),
                                  h4("stocks"), p("AAPL, NFLX, MSFT, AMZN  |  2010-01-01 to 2026-04-01"),
                                  hr(),
                                  h4("features"),
                                  tags$ul(
                                    tags$li("RSI (14-day)"),
                                    tags$li("stochastic oscillator: FastK, FastD, SlowD"),
                                    tags$li("Williams %R"),
                                    tags$li("MACD + signal line"),
                                    tags$li("price rate of change (14-day)"),
                                    tags$li("on-balance volume"),
                                    tags$li("rolling mean ratios: weekly, quarterly, annual"),
                                    tags$li("annual/weekly and annual/quarterly ratios"),
                                    tags$li("intraday ratios: open/close, high/close, low/close"),
                                    tags$li("k-means cluster label (k=3, from PCA scores)")
                                  ),
                                  hr(),
                                  h4("models"),
                                  tags$ul(
                                    tags$li(strong("logistic regression"), " — interpretable baseline"),
                                    tags$li(strong("random forest"), " — ensemble, 300 trees"),
                                    tags$li(strong("xgboost"), " — gradient boosting")
                                  ),
                                  hr(),
                                  h4("evaluation"),
                                  p("80/20 time-based split + 5-fold CV.
               metrics: accuracy, sensitivity, specificity, precision, recall, F1, AUC.")
                              )
                       )
                     )
           )
  )
)

server <- function(input, output, session) {
  
  #overview tab
  output$ov_bar <- renderPlot({
    req(length(all_metrics) > 0)
    df <- map_dfr(stocks, ~ all_metrics[[.x]] %>% mutate(ticker = .x))
    col_sel <- input$ov_metric
    ggplot(df, aes(x = ticker, y = .data[[col_sel]], fill = model)) +
      geom_col(position = "dodge", width = 0.7) +
      geom_text(aes(label = round(.data[[col_sel]], 3)),
                position = position_dodge(0.7),
                vjust = -0.4, size = 3.2, fontface = "bold") +
      scale_fill_manual(values = c("#1976D2","#43A047","#E53935")) +
      scale_y_continuous(limits = c(0, 1.1)) +
      labs(x = "ticker", y = col_sel, fill = "") +
      theme_bw(base_size = 12) +
      theme(legend.position = "bottom",
            panel.grid.major.x = element_blank())
  })
  
  output$ov_tbl <- renderDT({
    req(length(all_metrics) > 0)
    df <- map_dfr(stocks, ~ all_metrics[[.x]] %>% mutate(ticker = .x)) %>%
      select(ticker, model, everything())
    datatable(df, rownames = FALSE,
              options = list(pageLength = 12, dom = "tp")) %>%
      formatRound(c("accuracy","sensitivity","specificity",
                    "precision","recall","f1","auc"), 4)
  })
  
  #reactive ticker selection
  sel_t <- reactive(input$sel_ticker)
  
  #metrics table for selected ticker
  output$stock_metrics_tbl <- renderDT({
    req(all_metrics[[sel_t()]])
    datatable(all_metrics[[sel_t()]], rownames = FALSE,
              options = list(dom = "t")) %>%
      formatRound(c("accuracy","sensitivity","specificity",
                    "precision","recall","f1","auc"), 4)
  })
  
  #roc curves
  output$roc_plot <- renderPlot({
    req(all_roc_data[[sel_t()]])
    ggplot(all_roc_data[[sel_t()]],
           aes(x = fpr, y = tpr, color = model)) +
      geom_line(linewidth = 1.2) +
      geom_abline(linetype = "dashed", color = "gray60") +
      scale_color_manual(values = c("#1976D2","#43A047","#E53935")) +
      labs(title = paste("roc curves -", sel_t()),
           x = "false positive rate", y = "true positive rate", color = "") +
      theme_bw(base_size = 13) +
      theme(legend.position = "bottom",
            plot.title = element_text(face = "bold"))
  })
  
  #confusion matrices
  output$cm_lr_plot <- renderPlot({
    req(all_cm_data[[sel_t()]])
    draw_cm(all_cm_data[[sel_t()]]$lr,
            paste("LR -", sel_t()))
  })
  output$cm_rf_plot <- renderPlot({
    req(all_cm_data[[sel_t()]])
    draw_cm(all_cm_data[[sel_t()]]$rf,
            paste("RF -", sel_t()))
  })
  output$cm_xgb_plot <- renderPlot({
    req(all_cm_data[[sel_t()]])
    draw_cm(all_cm_data[[sel_t()]]$xgb,
            paste("XGB -", sel_t()))
  })
  

  
  #features tab reactive
  feat_t <- reactive(input$feat_ticker)
  
  output$rf_imp_plot <- renderPlot({
    req(all_imp_rf[[feat_t()]])
    top <- head(all_imp_rf[[feat_t()]], input$n_imp)
    ggplot(top, aes(x = reorder(feature, Overall), y = Overall)) +
      geom_col(fill = "#43A047", width = 0.65) +
      coord_flip() +
      labs(title = paste("random forest -", feat_t()),
           x = "", y = "mean decrease gini") +
      theme_bw(base_size = 12) +
      theme(panel.grid.major.y = element_blank())
  })
  
  output$xgb_imp_plot <- renderPlot({
    req(all_imp_xgb[[feat_t()]])
    top <- head(all_imp_xgb[[feat_t()]], input$n_imp)
    ggplot(top, aes(x = reorder(feature, Overall), y = Overall)) +
      geom_col(fill = "#FB8C00", width = 0.65) +coord_flip() +
      labs(title = paste("xgboost -", feat_t()),
           x = "", y = "relative importance") +
      theme_bw(base_size = 12) +
      theme(panel.grid.major.y = element_blank())
  })
  
  output$lr_forest_plot <- renderPlot({
    req(all_coef_lr[[feat_t()]])
    df <- na.omit(all_coef_lr[[feat_t()]])
    ggplot(df, aes(x = OR, y = reorder(feature, OR), color = direction)) +
      geom_vline(xintercept = 1, linetype = "dashed",
                 color = "gray50", linewidth = 0.8) +
      geom_errorbarh(aes(xmin = OR_lower, xmax = OR_upper),
                     height = 0.35, linewidth = 0.9) +
      geom_point(size = 3.5) +scale_color_manual(values = c("positive" = "#1976D2",
                                    "negative" = "#E53935")) +
      labs(title = paste("lr odds ratio forest plot -", feat_t()),
           subtitle = "OR > 1 = associated with UP;  OR < 1 = associated with DOWN",
           x = "odds ratio  |  whiskers = 95% CI", y = "", color = "") +
      theme_bw(base_size = 12) +
      theme(legend.position = "bottom",
            plot.subtitle = element_text(color = "#777", size = 10))+xlim(0,5)
  })
  
  output$lr_coef_tbl <- renderDT({
    req(all_coef_lr[[feat_t()]])
    df <- all_coef_lr[[feat_t()]] %>%select(feature, coef, OR, OR_lower, OR_upper, direction) %>%
      rename("log-OR" = coef, "HR/OR" = OR,
             "lower 95% CI" = OR_lower, "upper 95% CI" = OR_upper)
    datatable(df, rownames = FALSE,
              options = list(pageLength = 10, dom = "tp")) %>%
      formatRound(c("log-OR","HR/OR","lower 95% CI","upper 95% CI"), 4) %>%
      formatStyle("direction",
                  backgroundColor = styleEqual(
                    c("positive","negative"), c("#e8f5e9","#ffebee")))
  })
  
  #pca + clustering tab reactive
  pca_t <- reactive(input$pca_ticker)
  
  pca_obj_r <- reactive({
    req(all_pca_km[[pca_t()]])
    all_pca_km[[pca_t()]]
  })
  
  scree_df_r <- reactive({
    pca_imp <- summary(pca_obj_r()$pca)$importance
    n_pc    <- min(10, ncol(pca_imp))
    data.frame(
      PC     = 1:n_pc,VarExp = pca_imp[2, 1:n_pc] * 100,
      CumVar = pca_imp[3, 1:n_pc] * 100
    )
  })
  
  scores_r <- reactive({
    df <- pca_obj_r()$scores
    df$cluster  <- pca_obj_r()$cluster
    pca3        <- pca_obj_r()$pca$x[, 3]
    df$PC3      <- pca3[1:nrow(df)]
    df
  })
  
  output$pca_scree <- renderPlot({
    df <- scree_df_r()
    ggplot(df, aes(x = PC, y = VarExp)) +
      geom_col(fill = "#1976D2", width = 0.65) +
      geom_text(aes(label = paste0(round(VarExp, 1), "%")),
                vjust = -0.4, size = 3.5) +
      labs(title = paste("variance per component -", pca_t()),
           x = "PC", y = "variance (%)") +
      theme_bw(base_size = 12)
  })
  
  output$pca_cumvar <- renderPlot({
    df <- scree_df_r()
    ggplot(df, aes(x = PC, y = CumVar)) +
      geom_line(color = "#E53935", linewidth = 1.2) +
      geom_point(color = "#E53935", size = 3) +
      geom_hline(yintercept = 80, linetype = "dashed", color = "gray60") +
      geom_text(aes(label = paste0(round(CumVar, 1), "%")),
                vjust = -0.6, size = 3.2) +
      labs(title = paste("cumulative variance -", pca_t()),
           x = "PC", y = "cumulative (%)") +
      scale_y_continuous(limits = c(0, 105)) +
      theme_bw(base_size = 12)
  })
  
  output$cluster_scatter <- renderPlot({
    df <- scores_r()
    ggplot(df, aes(x = PC1, y = PC2, color = cluster)) +
      geom_point(alpha = 0.45, size = 1.5) +
      stat_ellipse(linewidth = 0.9, linetype = "dashed") +
      scale_color_manual(values = c("#1976D2","#43A047","#E53935")) +
      labs(title = paste("pc1 vs pc2 clusters -", pca_t()),
           x = "PC1", y = "PC2", color = "cluster") +
      theme_bw(base_size = 12) +
      theme(legend.position = "bottom")
  })
  
  output$cluster_bar <- renderPlot({
    df <- scores_r()
    count_df <- df %>% count(cluster)
    ggplot(count_df, aes(x = cluster, y = n, fill = cluster)) +
      geom_col(width = 0.6, show.legend = FALSE) +
      geom_text(aes(label = n), vjust = -0.4,
                fontface = "bold", size = 5) +
      scale_fill_manual(values = c("#1976D2","#43A047","#E53935")) +
      labs(title = paste("cluster sizes -", pca_t()),
           x = "cluster", y = "count") +
      theme_bw(base_size = 13) +
      theme(panel.grid.major.x = element_blank())
  })
  
  output$cluster_pc13 <- renderPlot({
    df <- scores_r()
    ggplot(df, aes(x = PC1, y = PC3, color = cluster)) +
      geom_point(alpha = 0.45, size = 1.5) +
      stat_ellipse(linewidth = 0.9, linetype = "dashed") +
      scale_color_manual(values = c("#1976D2","#43A047","#E53935")) +
      labs(title = paste("pc1 vs pc3 -", pca_t()),
           x = "PC1", y = "PC3", color = "cluster") +
      theme_bw(base_size = 12) +
      theme(legend.position = "bottom")
  })
  
  output$cluster_response <- renderPlot({
    req(all_results[[pca_t()]])
    # use training set only, which has matching cluster labels
    train_df       <- all_results[[pca_t()]]$train
    train_df$cluster <- all_pca_km[[pca_t()]]$cluster
    
    ct <- train_df %>%
      count(cluster, Response) %>%
      group_by(cluster) %>%
      mutate(pct = n / sum(n) * 100)
    
    ggplot(ct, aes(x = cluster, y = pct, fill = Response)) +
      geom_col(position = "dodge", width = 0.6) +
      geom_text(aes(label = paste0(round(pct, 1), "%")),
                position = position_dodge(0.6),
                vjust = -0.4, size = 3.5) +
      scale_fill_manual(values = c("DOWN" = "#E53935", "UP" = "#43A047")) +
      scale_y_continuous(limits = c(0, 80)) +
      labs(title = paste("up vs down per cluster -", pca_t()),
           x = "cluster", y = "percentage (%)", fill = "") +
      theme_bw(base_size = 12) +
      theme(legend.position = "bottom",
            panel.grid.major.x = element_blank())
  })
}

shinyApp(ui = ui, server = server)
