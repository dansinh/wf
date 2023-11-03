#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Setup
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1.1 Upload packages
pacman::p_load(
  tidyverse, tidymodels,
  doMC, ranger, xgboost, lightgbm, bonsai, probably, themis,
  janitor, skimr, ggthemes, ggsci, gt, gtExtras
)

# 1.2 Util functions
tidymodels_prefer(quiet = FALSE)
options(scipen = 999)

# Function to set visualization custom theme
my_theme <- function() {
  theme_fivethirtyeight() +
    theme(
      plot.title = element_text(size = 12),
      plot.title.position = "plot",
      plot.subtitle = element_text(size = 11),
      axis.title = element_text(size = 12),
      axis.text.x = element_text(size = 10, face = "bold"),
      axis.text.y = element_text(size = 10, face = "bold"),
      strip.text = element_text(size = 10, face = "bold")
    )
}
theme_set(my_theme())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df <- read_csv("loan_data.csv") |>
  clean_names() |>
  select(
    customer_id = id, loan_amount = loan_amnt, loan_status,
    term, interest_rate = int_rate, purpose, installment,
    employment_length = emp_length, home_ownership,
    annual_income = annual_inc, residence_state = addr_state,
    dti, delinquency_2years = delinq_2yrs,
    months_since_delinquency = mths_since_last_delinq,
    min_date_credit_line = earliest_cr_line,
    open_accounts = open_acc, total_accounts = total_acc,
    revolving_balance = revol_bal
  ) |>
  filter(!is.na(term)) |>
  mutate(
    loan_status = case_when(
      loan_status == "Charged Off" ~ "bad",
      loan_status == "Default" ~ "bad",
      loan_status == "Late (16-30 days)" ~ "bad",
      loan_status == "Late (31-120 days)" ~ "bad",
      loan_status == "In Grace Period" ~ "bad",
      loan_status == "Current" ~ "good",
      loan_status == "Fully Paid" ~ "good"
    ),
    loan_status = factor(loan_status),
    customer_id = factor(customer_id),
    months_since_delinquency = if_else(
      is.na(months_since_delinquency),
      0, months_since_delinquency
    ),
    employment_length = parse_number(employment_length),
    employment_length= if_else(
      is.na(employment_length),
      1, employment_length
    ),
    home_ownership = case_when(
      home_ownership == "RENT" ~ "rent",
      home_ownership == "OWN" ~ "own",
      .default = "mortgage"
    ),
    min_date_credit_line = as_date(min_date_credit_line),
    min_date_credit_line = if_else(
      min_date_credit_line >= today(), today(), min_date_credit_line
    ),
    months_since_first_credit = interval(min_date_credit_line, today()) %/% months(1),
    loan_income_ratio = loan_amount / annual_income,
    installment_balance_ratio = installment / (revolving_balance + 1),
    open_account_ratio = open_accounts/total_accounts,
    total_earnings = employment_length * annual_income,
  ) |>
  select(-c(min_date_credit_line, residence_state)) |>
  mutate(across(.cols = where(is.character), .fns = factor))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Loan Classification Models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3.1 Data budgeting with case weights
set.seed(1)
df_split <- initial_split(data = df, prop = 0.8, strata = loan_status)
df_train <- training(df_split)
df_test <- testing(df_split)
df_folds <- vfold_cv(data = df_train, v = 5)

# Proportion of bad loans (4.1%)
df_train |>
  count(loan_status) |>
  adorn_totals(where = c("col")) |>
  mutate(pct = 100 * n / sum(n)) |>
  mutate(across(.cols = where(is.numeric), .fns = \(x) round(x, 1))) |>
  select(-Total) |>
  gt() |> gt_theme_538()

# 3.2 Recipe set
recipe_listing <- function() {
  base_recipe <- recipe(
    loan_status ~ loan_amount + purpose + term + interest_rate + installment,
    data = df_train
  ) |>
    step_other(purpose, other = "rest", threshold = 0.01) |>
    step_rose(loan_status, over_ratio = 1) |>
    step_dummy(all_nominal_predictors()) |>
    step_novel(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors()) |>
    step_zv(all_predictors())
  
  mid_recipe <- recipe(
    loan_status ~ loan_amount + purpose + term + interest_rate + installment +
      delinquency_2years + months_since_delinquency +
      months_since_first_credit + revolving_balance +
      open_accounts + total_accounts,
    data = df_train
  ) |>
    step_other(purpose, other = "rest", threshold = 0.01) |>
    step_rose(loan_status, over_ratio = 1) |>
    step_dummy(all_nominal_predictors()) |>
    step_novel(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors()) |>
    step_zv(all_predictors())
  
  full_recipe <- recipe(
    loan_status ~ loan_amount + purpose + term + interest_rate + installment +
      delinquency_2years + months_since_delinquency +
      months_since_first_credit + revolving_balance +
      open_accounts + total_accounts +
      annual_income + dti + employment_length + home_ownership,
    data = df_train
  ) |>
    step_other(purpose, other = "rest", threshold = 0.01) |>
    step_rose(loan_status, over_ratio = 1) |>
    step_dummy(all_nominal_predictors()) |>
    step_novel(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors()) |>
    step_zv(all_predictors())
  
  generative_recipe <- recipe(
    loan_status ~ loan_amount + purpose + term + interest_rate + installment +
      delinquency_2years + months_since_delinquency +
      months_since_first_credit + revolving_balance +
      open_accounts + total_accounts +
      annual_income + dti + employment_length + home_ownership +
      loan_income_ratio + installment_balance_ratio +
      open_account_ratio + total_earnings,
    data = df_train
  ) |>
    step_other(purpose, other = "rest", threshold = 0.01) |>
    step_rose(loan_status, over_ratio = 1) |>
    step_dummy(all_nominal_predictors()) |>
    step_novel(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors()) |>
    step_zv(all_predictors())
  
  preproc <- list(
    base_recipe = base_recipe,
    mid_recipe = mid_recipe,
    full_recipe = full_recipe,
    generative_recipe = generative_recipe
  )
  return(preproc)
}

# 3.3 Model set
model_listing <- function() {
  glm_spec <- logistic_reg(penalty = 0) |>
      set_engine("glm") |>
      set_mode("classification")

  rf_spec <- rand_forest(trees = 1000) |>
    set_engine("ranger") |>
    set_mode("classification")
  
  xgb_spec <- boost_tree(trees = 1000) |>
    set_engine("xgboost") |>
    set_mode("classification")
  
  lightgbm_spec <- boost_tree(trees = 1000) |>
    set_engine("lightgbm") |>
    set_mode("classification")
  
  models = list(glm_spec, rf_spec, xgb_spec, lightgbm_spec)
  return(models)
}

# 3.4 Workflow set
wf_set <- workflow_set(
  preproc = recipe_listing(),
  models = model_listing(),
  cross = TRUE
)

# 3.5 Workflow evaluation
registerDoMC(cores = detectCores())
class_metrics <- wf_set |>
  workflow_map(
    fn = "fit_resamples",
    resamples = df_folds,
    seed = 1, verbose = TRUE,
    metrics = metric_set(
      j_index, spec, sens, mn_log_loss,
      f_meas, precision, recall
    ),
    control = control_resamples(
      save_pred = FALSE,
      save_workflow = FALSE,
      parallel_over = "everything"
    )
  )
registerDoSEQ()

collect_metrics(class_metrics) |>
  select(wflow_id, .metric, mean) |>
  mutate(across(where(is.numeric), \(x) round(x, 2))) |>
  pivot_wider(id_cols = wflow_id, names_from = .metric, values_from = mean) |>
  arrange(desc(j_index)) |>
  select(
    wflow_id, j_index, spec, sens,
    precision, recall, f_meas, mn_log_loss
  ) |>
  rename(
    `Recipe/Model Combo` = wflow_id,
    `J-Index` = j_index,
    `Log Loss` = mn_log_loss,
    `Specificity` = spec,
    `Sensitivity` = sens,
    `Precision` = precision,
    `Recall` = recall,
    `F1 Score` = f_meas
  ) |>
  gt() |> gt_theme_538() |>
  tab_header(
    title = 'Loan Classification - Model Screening',
    subtitle = "5-fold CV on training set (80%)"
  )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Final Model Evaluation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4.1 Logistic model predictions
generative_recipe <- recipe(
  loan_status ~ loan_amount + purpose + term + interest_rate + installment +
    delinquency_2years + months_since_delinquency +
    months_since_first_credit + revolving_balance +
    open_accounts + total_accounts +
    annual_income + dti + employment_length + home_ownership +
    loan_income_ratio + installment_balance_ratio +
    open_account_ratio + total_earnings,
  data = df_train
) |>
  step_other(purpose, other = "rest", threshold = 0.01) |>
  step_rose(loan_status, over_ratio = 1) |>
  step_dummy(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())

glm_spec <- logistic_reg(penalty = 0) |>
  set_engine("glm") |>
  set_mode("classification")

test_pred <- workflow() |>
  add_recipe(generative_recipe) |>
  add_model(glm_spec) |>
  fit(df_train) |>
  predict(df_test, type = "prob")

loan_predictions <- df_test |> bind_cols(test_pred) |>
  mutate(
    pred_loan_status = if_else(.pred_good >= .pred_bad, "good", "bad"),
    pred_loan_status = factor(pred_loan_status)
  )

# 4.3 Final predictions evaluation
class_metrics <- metric_set(j_index, sensitivity, specificity)
loan_predictions |>
  class_metrics(truth = loan_status, estimate = pred_loan_status) |>
  select(metric = .metric, estimate = .estimate) |>
  mutate(across(where(is.numeric), \(x) round(x, 2))) |>
  gt() |>
  gt_theme_538()

conf_mat(loan_predictions, truth = loan_status, estimate = pred_loan_status)

# 4.4 Save workspace
save.image(file = "wf_model_screening.RData")
