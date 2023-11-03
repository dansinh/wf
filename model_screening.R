#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Setup
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1.1 Upload packages
pacman::p_load(
  tidyverse, tidymodels, textrecipes, text2vec, stringr,
  doMC, ranger, xgboost, lightgbm, bonsai, finetune,
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

# Function to examine distributions of categorical variables
cumu_pct <- function(data, threshold, ...) {
  data |> count(..., sort = TRUE) |>
    mutate(percent = 100 * n / sum(n)) |>
    arrange(desc(percent)) |>
    mutate(cumu_percent = cumsum(percent)) |>
    filter(cumu_percent <= {{threshold}})
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data <- read_csv("loan_data.csv") |>
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
    is_bad_loan = case_when(
      loan_status == "Charged Off" ~ 1,
      loan_status == "Default" ~ 1,
      loan_status == "Late (16-30 days)" ~ 1,
      loan_status == "Late (31-120 days)" ~ 1,
      loan_status == "Current" ~ 0,
      loan_status == "Fully Paid" ~ 0,
      loan_status == "In Grace Period" ~ 0
    ),
    is_bad_loan = factor(is_bad_loan),
    customer_id = factor(customer_id),
    months_since_delinquency = if_else(
      is.na(months_since_delinquency),
      0, months_since_delinquency
    ),
    employment_length = parse_number(employment_length),
    employment_length= if_else(
      is.na(employment_length),
      mean(employment_length),
      employment_length
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
    months_since_first_credit = interval(min_date_credit_line, today()) %/% months(1)
  ) |>
  select(-c(min_date_credit_line, loan_status)) |>
  mutate(across(.cols = where(is.character), .fns = factor))

data |> skim()
data |> count(purpose)
data |> count(is_bad_loan)

cumu_pct(data = data, threshold = 100, purpose)

data |> count(purpose, sort = TRUE) |>
  mutate(percent = 100 * n / sum(n)) |>
  arrange(desc(percent)) |>
  mutate(cumu_percent = cumsum(percent))

# How many individuals/id's: 10k distinct
data |> summarize(n = n(), .by = id) |>
  arrange(desc(n))

# Missing data: 476 ~ 0.48% completely missing with only requested & provided

# Fixed or floating interest rate: fixed
data |> summarize(n_rate = n_distinct(int_rate), .by = id) |>
  arrange(desc(n_rate))

# Fixed or floating monthly installment: fixed
data |> summarize(n_install = n_distinct(installment), .by = id) |>
  arrange(desc(n_install))

# Dist of term of the loan: 72.7% (36m), 22.6% (60m), 0.5% (missing)
data |> count(term)

# Distribution of emp_length: 11 categories + NA: turn into numeric
data |> count(emp_length)

# Dist of home ownership: 48% mortgage, 39% rent, 8% own, group others into mortgage
data |> count(home_ownership) |> arrange(desc(n))

# Dist of loan status: target variable (likelihood of repayment)
# good: current (81.2%) + fully paid (9.5%)
# problematic: in grace (0.48%)
# bad: charged off (2.1%) + default (0.16%) + late (0.21%) + very late (1.48%)
data |> count(loan_status)
data |> filter(loan_status == "In Grace Period") |>
  summary()

# Purpose of the loan: 13, step_other by distribution
data |> count(purpose)

# Address state: 45
data |> count(addr_state)

# delinquency 2 years: 0-11 times
data |> count(delinq_2yrs)

# Revolving credit line/credit card balance: fixed credit line
data |> summarize(n_line = n_distinct(revol_bal), .by = id) |>
  arrange(desc(n_line))

# Dist of loan amount: 1k - 35k, funded: 1k - 35k
data |> summary()
data |> filter(!is.na(term)) |> summary()
data |> filter(!is.na(term)) |> slice_sample(n = 4) |> data.frame()
