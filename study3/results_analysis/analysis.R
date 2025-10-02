options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
packages <- c("tidyverse", "tableone", "lme4", "lmerTest", "broom",
              "broom.mixed", "minpack.lm", "interactions", "emmeans",
              "ggplot2", "forcats", "afex", "bruceR", "car", "knitr",
              "kableExtra", "sandwich", "effectsize")
for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
        install.packages(pkg)
    }
}
## ---------- A. LOG HELPERS ----------
get_script_path <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  m <- grep("--file=", ca)
  if (length(m) > 0) return(normalizePath(sub("^--file=", "", ca[m])))
  if (!is.null(sys.frames()[[1]]$ofile)) return(normalizePath(sys.frames()[[1]]$ofile))
  normalizePath(".")
}
script_path <- get_script_path()
script_dir  <- dirname(script_path)
setwd(script_dir)

emm_pick_stat <- function(tb) {
  nm <- intersect(c("t.ratio", "z.ratio", "ratio"), names(tb))
  if (length(nm) == 0) {
    list(values = rep(NA_real_, nrow(tb)), type = NA_character_)
  } else {
    list(values = tb[[nm[1]]],
         type   = dplyr::case_when(
           nm[1] == "t.ratio" ~ "t",
           nm[1] == "z.ratio" ~ "z",
           TRUE               ~ "ratio"
         ))
  }
}

emm_pick_ci <- function(tb) {
  lcl <- if ("lower.CL"  %in% names(tb)) tb$lower.CL  else if ("asymp.LCL" %in% names(tb)) tb$asymp.LCL else NA_real_
  ucl <- if ("upper.CL"  %in% names(tb)) tb$upper.CL  else if ("asymp.UCL" %in% names(tb)) tb$asymp.UCL else NA_real_
  list(lower = lcl, upper = ucl)
}

packages <- c(
  "tidyverse", "tableone", "lme4", "lmerTest", "broom",
  "broom.mixed", "minpack.lm", "interactions", "emmeans",
  "ggplot2", "forcats", "afex", "bruceR", "car",
  "knitr", "kableExtra", "sandwich", "effectsize"
)

installed <- rownames(installed.packages())
to_install <- setdiff(packages, installed)

if (length(to_install) > 0) {
  message("Installing missing packages: ", paste(to_install, collapse = ", "))
  install.packages(to_install, dependencies = TRUE)
}

suppressPackageStartupMessages({
  invisible(sapply(packages, require, character.only = TRUE))
})

options(
  dplyr.summarise.inform = FALSE,   
  width = 120                      
)
afex_options(type = 3)
suppressMessages({
  emm_options(
    lmerTest.limit   = 1e7,
    pbkrtest.limit   = 1e7,
    disable.pbkrtest = TRUE,
    rg.limit         = 20000,
    opt.digits       = 3
  )
})

options(warn = -1)

log_time <- function() format(Sys.time(), "%H:%M:%S")
log_info <- function(...)    message(sprintf("[%s] ✅ ", log_time()), paste0(..., collapse=""))
log_step <- function(...)    message(sprintf("[%s] →  ", log_time()), paste0(..., collapse=""))
log_note <- function(...)    message(sprintf("[%s] ℹ️  ", log_time()), paste0(..., collapse=""))
log_warn <- function(...)    message(sprintf("[%s] ⚠️  ", log_time()), paste0(..., collapse=""))
log_done <- function(...)    message(sprintf("[%s] 🎯 ", log_time()), paste0(..., collapse=""))

log_info("Working directory: ", getwd())

## ---------- B. PIPELINE HELPERS ----------
read_data <- function(path) {
  log_step("Loading data: ", path)
  read.csv(path, check.names = FALSE)
}

prep_group_factors <- function(df) {
  log_step("Preparing for analysis...")
  df %>%
    mutate(
      nudgeType = case_when(
        group == "control" ~ "control",
        grepl("_", group)  ~ sub("(_.*)$", "", group),
        TRUE               ~ NA_character_
      ),
      nudgeIntensity = case_when(
        group == "control" ~ "control",
        grepl("_", group)  ~ sub("^(.*)_", "", group),
        TRUE               ~ NA_character_
      )
    ) %>%
    mutate(
      nudgeIntensity = factor(nudgeIntensity, levels = c("control","null","low","high"), ordered = TRUE),
      nudgeType = factor(nudgeType),
      group     = factor(group),
      ID        = factor(ID)
    ) %>%
    mutate(
      nudgeType = stats::relevel(nudgeType, ref = "control"),
      group     = stats::relevel(group,     ref = "control")
    )
}

prep_demographics <- function(df) {
  df$ethnicity <- factor(df$ethnicity, levels = c("White","Black","Asian","Mixed","Other"))
  df$Sex <- factor(df$Sex, levels = c("Male","Female"))
  df %>%
    mutate(
      Nationality = factor(Nationality),
      occupation  = factor(occupation),
      income = factor(income, levels = 1:7, ordered = TRUE,
                      labels = c("Below £10,000","£10,001 – £20,000","£20,001 – £30,000",
                                 "£30,001 – £40,000","£40,001 – £60,000","Above £60,000",
                                 "Prefer not to say")),
      education = factor(education, levels = 1:7, ordered = TRUE,
                         labels = c("Primary school or below","Middle school","High school",
                                    "Associate degree","Bachelor’s degree","Master’s degree","Doctoral degree"))
    ) %>%
    mutate(
      education_grouped = factor(
        case_when(
          education %in% c("Primary school or below","Middle school","High school") ~ "High school or below",
          education == "Associate degree"   ~ "Associate degree",
          education == "Bachelor’s degree"  ~ "Bachelor’s degree",
          education %in% c("Master’s degree","Doctoral degree") ~ "Master or above",
          TRUE ~ NA_character_
        ),
        levels = c("High school or below","Associate degree","Bachelor’s degree","Master or above"),
        ordered = TRUE
      )
    ) %>%
    mutate(education = education_grouped) %>%
    select(-education_grouped)
}

build_day_level <- function(df, vars_baseline) {
  log_step("Building day-level long data...")
  willing_cols  <- grep("^scenario\\d+_willingness_day\\d+$", names(df), value = TRUE)
  behav_cols    <- gsub("willingness", "behavior", willing_cols)
  
  long_will <- df %>%
    select(ID, all_of(willing_cols)) %>%
    pivot_longer(cols = all_of(willing_cols),
                 names_to = c("scenario","day"),
                 names_pattern = "scenario(\\d+)_willingness_day(\\d+)",
                 values_to = "willingness")
  
  long_beh <- df %>%
    select(ID, all_of(behav_cols)) %>%
    pivot_longer(cols = all_of(behav_cols),
                 names_to = c("scenario","day"),
                 names_pattern = "scenario(\\d+)_behavior_day(\\d+)",
                 values_to = "behavior")
  
  long <- long_beh %>%
    left_join(long_will, by = c("ID","scenario","day")) %>%
    mutate(day = as.integer(day), scenario = factor(scenario))
  
  df_day <- long %>%
    group_by(ID, day) %>%
    summarise(
      mean_behavior    = mean(behavior, na.rm = TRUE),
      mean_willingness = mean(willingness, na.rm = TRUE),
      .groups = "drop"
    )
  
  df_day %>%
    left_join(df %>% select(ID, group, nudgeIntensity, nudgeType, all_of(vars_baseline)),
              by = "ID") %>%
    mutate(day_c = day - 1)
}

pairwise_vs_control_within_day <- function(df_day, outcome, day_pair, caption = NULL) {
  lab <- paste0("Day", day_pair)
  log_step("EMMs: ", outcome, " – within-day contrasts vs control (", lab[1], " / ", lab[2], ")")
  
  dat <- df_day %>%
    dplyr::filter(day %in% day_pair) %>%
    dplyr::mutate(day_f = factor(day, levels = day_pair, labels = lab))
  
  fml <- as.formula(sprintf(
    "%s ~ day_f * group + Age + ethnicity + Sex + Nationality + occupation + income + education + envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre + (1|ID)",
    outcome
  ))
  
  fit <- lmer(fml, data = dat)
  
  emm_by_day <- emmeans(
    fit, ~ group | day_f,
    nuisance   = c("Sex","ethnicity","Nationality","occupation","income","education"),
    cov.reduce = function(x) if (is.numeric(x)) mean(x, na.rm=TRUE) else x[1],
    weights    = "proportional"
  )
  
  contr <- contrast(emm_by_day, method = "trt.vs.ctrl", ref = "control", by = "day_f")
  out_raw <- summary(contr, infer = TRUE, adjust = "BH") %>% as_tibble()
  out_raw <- summary(contr, infer = TRUE, adjust = "BH") %>% as_tibble()
  
  stat <- emm_pick_stat(out_raw)
  ci   <- emm_pick_ci(out_raw)
  
  out <- out_raw %>%
    dplyr::mutate(
      test_stat = stat$values,
      stat_type = stat$type,
      df_out    = if ("df" %in% names(out_raw)) df else NA_real_,
      LCL       = ci$lower,
      UCL       = ci$upper
    ) %>%
    dplyr::transmute(
      day       = day_f,
      group     = contrast,
      estimate, SE,
      test_stat, 
      df = df_out,
      p      = p.value,
      lower.CL  = LCL,
      upper.CL  = UCL
    )
  log_info("EMMs: ", outcome, " – within-day contrasts vs control (", lab[1], " / ", lab[2], ") completed.")
  print(kable(out, digits = 3))
  invisible(out)
}

event_window_low <- function(df_day) {
  log_step("Event-window analysis (low intensity arms)")
  low_data <- df_day %>%
    filter(nudgeIntensity == "low") %>%
    mutate(
      event_window = case_when(
        day %in% c(2,5,8,11,14)  ~ "NudgeDay",
        day %in% c(3,6,9,12,15)  ~ "PostDay1",
        day %in% c(4,7,10,13,16) ~ "PostDay2",
        TRUE ~ NA_character_
      )
    ) %>%
    filter(!is.na(event_window)) %>%
    mutate(event_window = factor(event_window, levels = c("NudgeDay","PostDay1","PostDay2")))
  
  fit <- lmer(
    mean_behavior ~ event_window * nudgeType +
      Age + ethnicity + Sex + Nationality + occupation + income + education +
      envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre +
      (1 | ID),
    data = low_data, REML = FALSE
  )
  emm <- emmeans(
    fit, ~ event_window | nudgeType,
    nuisance = c("Sex","ethnicity","Nationality","occupation","income","education"),
    cov.reduce = function(x) if(is.numeric(x)) mean(x, na.rm=TRUE) else x[1],
    weights = "proportional"
  )
  contr <- contrast(
    emm,
    method = list("NudgeDay - PostDay1" = c(1,-1,0), "NudgeDay - PostDay2" = c(1,0,-1)),
    by = "nudgeType"
  )
  res <- summary(contr, infer = TRUE, adjust = "BH")
  log_info("Event-window contrasts computed.")
  res <- res %>% dplyr::select(-df)
  print(kable(res, digits = 3))
  invisible(res)
}

print_std_beta_table_hlmsummary <- function(mod, outcome_var = "mean_behavior", verbose = FALSE) {
  raw <- capture.output(bruceR::HLM_summary(mod))
  if (length(raw) == 0) {
    message("No output captured from HLM_summary(mod).")
    return(invisible(NULL))
  }
  
  i1 <- which(grepl("Standardized Coefficients", raw, fixed = TRUE))[1]
  if (is.na(i1)) {
    message("HLM_summary block not found (no line contains 'Standardized Coefficients').")
    if (verbose) cat(paste(raw, collapse = "\n"))
    return(invisible(NULL))
  }
  
  seg <- raw[(i1 + 1):length(raw)]
  
  is_rule <- function(x) grepl("^\\s*[-─]{5,}\\s*$", x)
  
  while (length(seg) > 0 && trimws(seg[1]) == "") seg <- seg[-1]
  if (length(seg) == 0) {
    message("No table after the heading.")
    return(invisible(NULL))
  }
  
  r1 <- which(is_rule(seg))[1]
  if (!is.na(r1)) seg2 <- seg[(r1 + 1):length(seg)] else seg2 <- seg
  
  r2 <- which(is_rule(seg2))[1]
  if (!is.na(r2)) seg3 <- seg2[(r2 + 1):length(seg2)] else seg3 <- seg2
  
  stop_idx <- which(seg3 == "" | is_rule(seg3))
  if (length(stop_idx)) {
    data_lines <- seg3[seq_len(stop_idx[1] - 1)]
  } else {
    data_lines <- seg3
  }
  
  data_lines <- data_lines[nzchar(trimws(data_lines))]
  if (!length(data_lines)) {
    message("Coefficient rows not detected (empty after trimming).")
    if (verbose) {
      cat("---- RAW BLOCK ----\n")
      cat(paste(seg, collapse = "\n"), "\n")
    }
    return(invisible(NULL))
  }
  
  pattern <- "^(.+?)\\s+(-?\\d+\\.?\\d*)\\s*\\((-?\\d+\\.?\\d*)\\)\\s+(-?\\d+\\.?\\d*)\\s+(\\d+\\.?\\d*)\\s+([^\\[]+)\\[\\s*(-?\\d+\\.?\\d*)\\s*,\\s*(-?\\d+\\.?\\d*)\\s*\\]$"
  
  parse_line <- function(x) {
    m <- regexec(pattern, x)
    a <- regmatches(x, m)[[1]]
    if (length(a) == 0) return(NULL)
    data.frame(
      term = trimws(a[2]),
      beta = as.numeric(a[3]),
      SE   = as.numeric(a[4]),
      t    = as.numeric(a[5]),
      df   = as.numeric(a[6]),
      p    = trimws(a[7]),
      `95% CI of β` = sprintf("[%.3f, %.3f]", as.numeric(a[8]), as.numeric(a[9])),
      stringsAsFactors = FALSE
    )
  }
  
  rows <- do.call(rbind, lapply(data_lines, parse_line))
  if (is.null(rows) || !nrow(rows)) {
    message("Parse failed — the table layout may differ on your system.")
    if (verbose) {
      cat("---- DATA LINES ----\n")
      cat(paste(data_lines, collapse = "\n"), "\n")
    }
    return(invisible(NULL))
  }
  
  cat("\nStandardized Coefficients (β):\n")
  cat(sprintf("Outcome Variable: %s\n", outcome_var))
  print(knitr::kable(rows, digits = 3, align = c("l","r","r","r","r","l","c")))
  invisible(rows)
}

attrition_summary <- function(df) {
  log_step("Attrition: computing follow-up completion rates")
  df_long <- df %>%
    pivot_longer(
      cols      = matches("^scenario\\d+_willingness_day\\d+$"),
      names_to  = c("scenario","day"),
      names_pattern = "scenario(\\d+)_willingness_day(\\d+)",
      values_to = "willingness"
    ) %>%
    mutate(day = as.integer(day), valid = !is.na(willingness)) %>%
    group_by(ID, group, day) %>%
    summarise(valid_any = any(valid), .groups = "drop")
  
  n_total <- df %>% distinct(ID) %>% nrow()
  
  completed_any_followup <- df_long %>%
    filter(day >= 2 & day <= 16, valid_any) %>%
    distinct(ID) %>% nrow()
  pct_any <- round(completed_any_followup / n_total * 100, 2)
  
  completed_day16 <- df_long %>%
    filter(day == 16, valid_any) %>%
    distinct(ID) %>% nrow()
  pct_d16 <- round(completed_day16 / n_total * 100, 2)
  
  log_info("Follow-up ≥1 day: ", completed_any_followup, " / ", n_total, " (", pct_any, "%)")
  log_info("Completed Day 16: ", completed_day16,       " / ", n_total, " (", pct_d16, "%)")
  
  tibble(
    metric = c("Follow-up ≥1 day", "Completed Day 16"),
    n      = c(completed_any_followup, completed_day16),
    N      = n_total,
    pct    = c(pct_any, pct_d16)
  )
}

within_group_change <- function(df_day, outcome, day_pair) {
  lab <- paste0("Day", day_pair)
  log_step("EMMs: ", outcome, " – within-group change (", lab[1], " → ", lab[2], ")")
  
  dat <- df_day %>%
    filter(day %in% day_pair) %>%
    mutate(day_f = factor(day, levels = day_pair, labels = lab))
  
  fml <- as.formula(sprintf(
    "%s ~ day_f * group + Age + ethnicity + Sex + Nationality + occupation + income + education + envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre + (1|ID)",
    outcome
  ))
  
  fit <- lmer(fml, data = dat)
  
  emm_day_by_group <- emmeans(
    fit, ~ day_f | group,
    nuisance   = c("Sex","ethnicity","Nationality","occupation","income","education"),
    cov.reduce = function(x) if (is.numeric(x)) mean(x, na.rm=TRUE) else x[1],
    weights    = "proportional"
  )
  contr <- contrast(emm_day_by_group, method = "revpairwise", adjust = "BH")
  out_raw <- summary(contr, infer = TRUE, adjust = "BH") %>% as_tibble()
  
  stat <- emm_pick_stat(out_raw)
  ci   <- emm_pick_ci(out_raw)
  
  out <- out_raw %>%
    mutate(
      test_stat = stat$values,
      stat_type = stat$type,
      df_out    = if ("df" %in% names(out_raw)) df else NA_real_,
      LCL       = ci$lower,
      UCL       = ci$upper
    ) %>%
    transmute(
      group, contrast, estimate, SE, 
      test_stat, p = p.value, 
      lower.CL = LCL, upper.CL = UCL
    )
  log_info("EMMs: ", outcome, " – within-group change (", lab[1], " → ", lab[2], ") completed.")
  print(kable(out, digits = 3))
  invisible(out)
}

fit_exp_decay_vs_linear <- function(df_day, outcome) {
  log_step("Fitting nonlinear (exp) vs linear models for ", outcome)
  dat <- df_day %>% dplyr::filter(day > 1)
  
  group_daily <- dat %>%
    dplyr::group_by(group, day) %>%
    dplyr::summarise(y = mean(.data[[outcome]], na.rm = TRUE), .groups = "drop") %>%
    dplyr::arrange(group, day)
  
  get_start_vals <- function(d) {
    y0 <- d$y[d$day == 2][1]; if (is.na(y0)) y0 <- 0.5
    asym <- if (all(is.na(d$y))) 0 else min(d$y, na.rm = TRUE)
    list(y0 = y0, asym = asym, k = 0.2)
  }
  
  lb_tight <- c(y0 = 0.0,  asym = -0.5, k = 1e-4)
  ub_tight <- c(y0 = 1.2,  asym =  1.2, k = 5.0)
  lb_loose <- c(y0 = -1.0, asym = -1.0, k = 1e-6)
  ub_loose <- c(y0 =  2.0, asym =  2.0, k = 10.0)
  
  attempt_nls <- function(df_one, start0, outcome) {
    k_starts <- c(start0$k, 0.05, 0.1, 0.2, 0.5)
    try_once <- function(bounds = c("tight","loose","none")) {
      for (k0 in unique(k_starts)) {
        st <- list(y0 = start0$y0, asym = start0$asym, k = k0)
        fit <- tryCatch({
          if (bounds[1] == "none") {
            nlsLM(y ~ asym + (y0 - asym) * exp(-k * (day - 2)),
                  data = df_one, start = st, control = nls.lm.control(maxiter = 500))
          } else if (bounds[1] == "tight") {
            nlsLM(y ~ asym + (y0 - asym) * exp(-k * (day - 2)),
                  data = df_one, start = st,
                  lower = lb_tight, upper = ub_tight,
                  control = nls.lm.control(maxiter = 500))
          } else {
            nlsLM(y ~ asym + (y0 - asym) * exp(-k * (day - 2)),
                  data = df_one, start = st,
                  lower = lb_loose, upper = ub_loose,
                  control = nls.lm.control(maxiter = 500))
          }
        }, error = function(e) NULL)
        if (!is.null(fit)) return(fit)
      }
      NULL
    }
    
    if (identical(outcome, "mean_behavior")) {
      fit <- try_once("tight"); if (!is.null(fit)) return(fit)
      fit <- try_once("loose"); if (!is.null(fit)) return(fit)
      fit <- try_once("none");  if (!is.null(fit)) return(fit)
    } else {
      fit <- try_once("loose"); if (!is.null(fit)) return(fit)
      fit <- try_once("none");  if (!is.null(fit)) return(fit)
      fit <- try_once("tight"); if (!is.null(fit)) return(fit)
    }
    NULL
  }
  
  calc_R2 <- function(obs, fit_aug) {
    if (is.null(fit_aug) || nrow(fit_aug) == 0) return(NA_real_)
    stats::cor(obs, fit_aug$.fitted, use = "complete.obs")^2
  }
  calc_auc <- function(x, y) {
    if (length(x) < 2) return(NA_real_)
    ord <- order(x); x2 <- x[ord]; y2 <- y[ord]
    sum(diff(x2) * (head(y2, -1) + tail(y2, -1)) / 2)
  }
  
  fits_tbl <- group_daily %>%
    dplyr::group_by(group) %>%
    tidyr::nest() %>%
    dplyr::mutate(
      start   = purrr::map(data, get_start_vals),
      fit_nl  = purrr::map2(data, start, ~ attempt_nls(.x, .y, outcome)),
      converged = purrr::map_lgl(fit_nl, ~ !is.null(.x)),
      coef = purrr::map(fit_nl, ~ {
        if (is.null(.x)) {
          tibble::tibble(term = c("y0","asym","k"), estimate = NA_real_)
        } else {
          broom::tidy(.x) |>
            dplyr::select(term, estimate) |>
            dplyr::filter(term %in% c("y0","asym","k"))
        }
      }),
      pred_nl = purrr::map2(fit_nl, data, ~ if (is.null(.x)) tibble::tibble() else broom::augment(.x, newdata = .y)),
      fit_lin  = purrr::map(data, ~ lm(y ~ day, data = .x)),
      pred_lin = purrr::map2(fit_lin, data, ~ broom::augment(.x, newdata = .y)),
      lin_coef = purrr::map(fit_lin, ~ broom::tidy(.x) %>% dplyr::filter(term == "day") %>% dplyr::transmute(k_lin = estimate))
    ) %>%
    dplyr::ungroup()
  
  not_conv <- fits_tbl %>% dplyr::filter(!converged) %>% dplyr::pull(group) %>% unique()
  
  metrics <- fits_tbl %>%
    dplyr::mutate(
      R2_nl   = purrr::map2_dbl(data, pred_nl,  ~ calc_R2(.x$y, .y)),
      R2_lin  = purrr::map2_dbl(data, pred_lin, ~ calc_R2(.x$y, .y)),
      delta_R2 = R2_nl - R2_lin
    ) %>%
    dplyr::select(group, R2_nl, R2_lin, delta_R2)
  
  coef_wide <- fits_tbl %>%
    dplyr::select(group, coef) %>% tidyr::unnest(coef) %>%
    dplyr::distinct(group, term, .keep_all = TRUE) %>%
    tidyr::complete(group, term = c("y0","asym","k")) %>%
    tidyr::pivot_wider(names_from = term, values_from = estimate)
  
  auc_tbl <- fits_tbl %>%
    dplyr::select(group, pred_nl) %>%
    dplyr::mutate(has_nl = purrr::map_lgl(pred_nl, ~ nrow(.x) > 1)) %>%
    dplyr::mutate(auc = purrr::map2_dbl(pred_nl, has_nl, ~ if (!.y) NA_real_ else calc_auc(.x$day, .x$.fitted))) %>%
    dplyr::select(group, auc)
  
  T_tbl <- fits_tbl %>%
    dplyr::transmute(
      group,
      T = purrr::map_dbl(seq_len(n()), ~ {
        pn <- pred_nl[[.x]]
        if (!is.null(pn) && "day" %in% names(pn) && length(pn$day) > 0) {
          return(max(pn$day, na.rm = TRUE) - 2)
        }
        d0 <- data[[.x]]
        if (!is.null(d0) && "day" %in% names(d0) && length(d0$day) > 0) {
          return(max(d0$day, na.rm = TRUE) - 2)
        }
        NA_real_
      })
    )
  
  mde_pr <- coef_wide %>%
    dplyr::left_join(T_tbl, by = "group") %>%
    dplyr::mutate(
      MDE = dplyr::case_when(
        is.na(y0) | is.na(asym) | is.na(k) | is.na(T) | T <= 0 ~ NA_real_,
        abs(k) < 1e-8 ~ y0,
        TRUE ~ asym + ((y0 - asym) / (k * T)) * (1 - exp(-k * T))
      ),
      PR = dplyr::case_when(is.na(y0) | is.na(asym) | y0 == 0 ~ NA_real_, TRUE ~ asym / y0)
    ) %>%
    dplyr::select(group, MDE, PR)
  
  k_lin_tbl <- fits_tbl %>% dplyr::select(group, lin_coef) %>% tidyr::unnest(lin_coef)
  
  res <- coef_wide %>%
    dplyr::left_join(auc_tbl,  by = "group") %>%
    dplyr::left_join(mde_pr,   by = "group") %>%
    dplyr::left_join(metrics,  by = "group") %>%
    dplyr::left_join(k_lin_tbl, by = "group") %>%
    dplyr::mutate(dplyr::across(where(is.numeric), ~ round(., 3))) %>%
    dplyr::select(group, k_lin, y0, asym, k, auc, MDE, PR, R2_nl, R2_lin, delta_R2)
  
  log_info("Nonlinear vs linear fit summary for ", outcome)
  print(kable(res, digits = 2))
  invisible(res)
}

compute_gap_and_lmm <- function(df_day) {
  log_step("Computing willingness–behaviour gap")
  df_day2 <- df_day %>%
    mutate(
      will_01 = (mean_willingness - 1) / 4,
      gap     = will_01 - mean_behavior
    )
  df_ml <- df_day2 %>%
    filter(nudgeIntensity %in% c("high","low","null")) %>%
    mutate(nudgeIntensity = factor(nudgeIntensity, levels = c("null","low","high"), ordered = TRUE),
           nudgeType = factor(nudgeType, levels = c('struc','assis','infor')))
  
  log_step("Fitting LMM for GAP ~ day_c * intensity * type + covariates")
  mod_gap <- lmer(
    gap ~ day_c * nudgeIntensity * nudgeType + 
      Age + ethnicity + Sex + Nationality + occupation + income + education +
      envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre +
      (1 | ID),
    data = df_ml
  )
  
  log_info("Main fixed effects from the linear mixed-effects model predicting the willingness--behaviour gap.")
  std_beta_tbl <- print_std_beta_table_hlmsummary(mod_gap, outcome_var = "gap")
  
  invisible(list(df_day = df_day2, df_ml = df_ml, model = mod_gap, beta_table = std_beta_tbl))
}

prepost_anova <- function(df, pre_var, post_var, label = "Outcome") {
  log_step("ANOVA (pre vs post) × group for ", label)
  df_long <- df %>%
    dplyr::select(ID, nudgeType, nudgeIntensity, group,
                  pre  = all_of(pre_var),
                  post = all_of(post_var)) %>%
    tidyr::pivot_longer(c(pre, post), names_to = "time", values_to = "value") %>%
    mutate(time = factor(time, levels = c("pre","post")))
  
  aov_obj <- aov_ez(
    id = "ID", dv = "value", data = df_long,
    within = "time", between = "group",
    anova_table = list(es = "pes")
  )
  print(aov_obj)
  emm <- emmeans(aov_obj, ~ time | group)
  log_info("ANOVA results for pre--post changes in ", label, ".\n", sep = "")
  prs <- summary(pairs(emm, adjust = "BH"), infer = TRUE)
  print(prs)
  
  mean_tab <- df_long %>%
    group_by(group, time) %>%
    summarise(
      n    = sum(!is.na(value)),
      mean = mean(value, na.rm = TRUE),
      sd   = sd(value, na.rm = TRUE),
      se   = sd / sqrt(n),
      ci_l = mean + qt(0.025, df = n - 1) * se,
      ci_u = mean + qt(0.975, df = n - 1) * se,
      min  = min(value, na.rm = TRUE),
      max  = max(value, na.rm = TRUE),
      .groups = "drop"
    ) %>% arrange(group, time)
  invisible(list(aov = aov_obj, pairs = prs, means = mean_tab))
}

internalisation_block <- function(df, vars_men_pre) {
  log_step("Descriptives for internalisation (pre/post)")
  vars_men_post <- gsub("_pre", "_post", vars_men_pre)
  df_men <- df %>% dplyr::select(all_of(vars_men_pre), all_of(vars_men_post)) %>%
    mutate(
      `attitude pre`       = envAttitude_pre,
      `attitude post`      = envAttitude_post,
      `self-efficacy pre`  = envSelfEfficacy_pre, 
      `self-efficacy post` = envSelfEfficacy_post,
      `motivation pre`     = envMotivation_pre,
      `motivation post`    = envMotivation_post
    ) %>%
    dplyr::select(`attitude pre`, `self-efficacy pre`,`motivation pre`,
                  `attitude post`,`self-efficacy post`,`motivation post`)
  log_info("Descriptive statistics for environmental autonomous motivation, environmental attitude, and environmental self-efficacy.\n")
  Describe(df_men)
  
  res_att <- prepost_anova(df, "envAttitude_pre",      "envAttitude_post",      "environmental attitude")
  res_mot <- prepost_anova(df, "envMotivation_pre",    "envMotivation_post",    "environmental autonomous motivation")
  res_se  <- prepost_anova(df, "envSelfEfficacy_pre",  "envSelfEfficacy_post",  "environmental self-efficacy")
  invisible(list(describe = df_men, attitude = res_att, motivation = res_mot, selfeff = res_se))
}

nudge_feedback_analyses <- function(df, p_adjust = "BH",
                                    emmeans_spec = c("intensity_by_type","type_only"),
                                    alpha = 0.05) {
  emmeans_spec <- match.arg(emmeans_spec)
  
  p_fmt <- function(p) ifelse(is.na(p), "NA",
                              ifelse(p < .001, "< .001", sprintf("= %.3f", p)))
  
  analyze_one <- function(yvar) {
    log_step('Performing regression on', yvar)
    cat("\n", strrep("=", 72), "\n", sep = "")
    cat(sprintf("Outcome: %s\n", yvar))
    cat(strrep("-", 72), "\n")
    
    fml <- as.formula(paste(yvar, "~ nudgeType * nudgeIntensity"))
    fit <- lm(fml, data = df)
    aov_res <- car::Anova(fit, type = 3)
    
    log_info("ANOVA for", yvar)
    print(aov_res)
    cat(strrep("-", 72), "\n")
    
    if (emmeans_spec == "intensity_by_type") {
      emm_obj <- emmeans::emmeans(fit, ~ nudgeIntensity | nudgeType)
    }
    else { # 
      emm_obj <- emmeans::emmeans(fit, ~ nudgeType)
    }
    
    log_info(
      "Pairwise comparisons for ", yvar,
      " (Estimated Marginal Means of ", 
      if (emmeans_spec == "intensity_by_type") "nudgeIntensity within each nudgeType" else "nudgeType overall",
      ", p-values adjusted by ", p_adjust, ").",
      sep = ""
    )
    pairs = summary(pairs(emm_obj, adjust = p_adjust), infer = TRUE)
    print(pairs)
    
    invisible(list(
      yvar      = yvar,
      aov       = aov_res,
      emm       = emm_obj,
      pairs = pairs
    ))
  }
  
  log_step("Descriptives & correlations for nudge feedback")
  vars_post <- c("nudge_acceptance_post","nudge_reactance_post","nudge_fatigue_post")
  df_post <- df %>%
    dplyr::select(dplyr::all_of(vars_post)) %>%
    dplyr::mutate(
      `nudge acceptance` = nudge_acceptance_post,
      `nudge reactance`  = nudge_reactance_post,  
      `nudge fatigue`    = nudge_fatigue_post
    ) %>%
    dplyr::select(`nudge acceptance`, `nudge reactance`, `nudge fatigue`)
  Describe(df_post)
  #Corr(df_post, plot = FALSE)
  
  out1 <- analyze_one("nudge_acceptance_post")
  out2 <- analyze_one("nudge_fatigue_post")
  out3 <- analyze_one("nudge_reactance_post")
  
  invisible(list(acceptance = out1, fatigue = out2, reactance = out3))
}

willingness_change_regressions <- function(df) {
  log_step("Computing Δwillingness and predictors (Δmotivation/Δattitude/acceptance)")
  mean_cols <- function(day) grep(paste0("^scenario\\d+_willingness_day", day, "$"), names(df), value = TRUE)
  df2 <- df %>%
    mutate(
      mean_will_day16 = rowMeans(dplyr::select(., all_of(mean_cols(16))), na.rm = TRUE),
      mean_will_day2  = rowMeans(dplyr::select(., all_of(mean_cols(2))),  na.rm = TRUE),
      delta_will      = mean_will_day16 - mean_will_day2,
      delta_motivation = envMotivation_post   - envMotivation_pre,
      delta_attitude   = envAttitude_post     - envAttitude_pre,
      delta_se         = envSelfEfficacy_post - envSelfEfficacy_pre
    )
  df_noco <- df2 %>%
    filter(group != "control") %>%
    mutate(nudgeType = factor(nudgeType, levels = c("struc","assis","infor"))) %>%
    droplevels()
  
  m_mot <- lm(delta_will ~ delta_motivation      * nudgeType, data = df_noco)
  m_att <- lm(delta_will ~ delta_attitude        * nudgeType, data = df_noco)
  m_acc <- lm(delta_will ~ nudge_acceptance_post * nudgeType, data = df_noco)
  
  get_trend_tbl <- function(model, pred, label){
    tr <- emtrends(model, ~ nudgeType, var = pred)
    as.data.frame(tr) %>%
      transmute(
        Predictor = label,
        NudgeType = nudgeType,
        Estimate  = !!sym(paste0(pred, ".trend")),
        SE, df,
        `Lower 95% CI` = lower.CL,
        `Upper 95% CI` = upper.CL,
        t = Estimate/SE,
        p = 2*pt(-abs(Estimate/SE), df = df)
      )
  }
  tbl_mot <- get_trend_tbl(m_mot, "delta_motivation",      "ΔMotivation")
  tbl_att <- get_trend_tbl(m_att, "delta_attitude",         "ΔAttitude")
  tbl_acc <- get_trend_tbl(m_acc, "nudge_acceptance_post",  "Nudge Acceptance")
  slopes_all_num <- bind_rows(tbl_mot, tbl_att, tbl_acc) %>% arrange(Predictor, NudgeType)
  
  cont_vars <- c("delta_will", "delta_motivation", "delta_attitude", "nudge_acceptance_post")
  df_noco_std <- df_noco %>% mutate(across(all_of(cont_vars), ~ as.numeric(scale(.))))
  m_mot_std <- lm(delta_will ~ delta_motivation      * nudgeType, data = df_noco_std)
  m_att_std <- lm(delta_will ~ delta_attitude        * nudgeType, data = df_noco_std)
  m_acc_std <- lm(delta_will ~ nudge_acceptance_post * nudgeType, data = df_noco_std)
  
  get_fmt <- function(model, pred, label){
    tr <- emtrends(model, ~ nudgeType, var = pred)
    as.data.frame(tr) %>%
      transmute(
        Predictor = label,
        NudgeType = nudgeType,
        Estimate  = !!sym(paste0(pred, ".trend")),
        SE, df,
        `Lower 95% CI` = lower.CL,
        `Upper 95% CI` = upper.CL,
        t = Estimate/SE,
        p = 2*pt(-abs(Estimate/SE), df = df)
      ) %>%
      mutate(
        Estimate = round(Estimate, 3),
        SE       = round(SE, 3),
        t        = round(t, 2),
        `Lower 95% CI` = round(`Lower 95% CI`, 3),
        `Upper 95% CI` = round(`Upper 95% CI`, 3),
        p = ifelse(is.na(p), NA, ifelse(p < 0.001, "< 0.001", formatC(p, format="f", digits=3)))
      )
  }
  tbl_mot_std <- get_fmt(m_mot_std, "delta_motivation",      "ΔMotivation")
  tbl_att_std <- get_fmt(m_att_std, "delta_attitude",         "ΔAttitude")
  tbl_acc_std <- get_fmt(m_acc_std, "nudge_acceptance_post",  "Nudge Acceptance")
  slopes_beta_fmt <- bind_rows(tbl_mot_std, tbl_att_std, tbl_acc_std) %>% arrange(Predictor, NudgeType)
  
  log_info("Standardized (β) regression estimates of predictors on willingness change (Day 16 - Day 2), by nudge type.\n")
  print(slopes_beta_fmt)
  invisible(list(models = list(m_mot = m_mot, m_att = m_att, m_acc = m_acc),
                 slopes_num = slopes_all_num, slopes_beta = slopes_beta_fmt,
                 data = list(df = df2, df_noco = df_noco, df_noco_std = df_noco_std)))
}

lmm_will_block <- function(df_day, verbose = FALSE) {
  log_step("Fitting LMM for mean_willingness (exclude control; random slope day_c by ID)")
  
  df_ml <- df_day |>
    dplyr::filter(day > 1)
  
  lmm_will <- suppressWarnings(lmer(
    mean_willingness ~ day_c * nudgeIntensity * nudgeType +
      Age + ethnicity + Sex + Nationality + occupation + income + education +
      envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre +
      (day_c | ID),
    data = df_ml, REML = FALSE
  ))
  
  std_beta_tbl <- print_std_beta_table_hlmsummary(lmm_will, outcome_var = "mean_willingness")
  
  log_info("LMM (mean_willingness) completed.")
  invisible(list(model = lmm_will, beta_table = std_beta_tbl))
}

rt_block <- function(df, df_day) {
  log_step("Computing RT metrics")
  
  std_beta_lm <- function(formula, data, focal = NULL, note = NULL) {
    resp <- all.vars(formula)[1]
    preds <- attr(terms(formula), "term.labels")
    if (length(preds) == 0) {
      log_warn("Skip std_beta_lm — no predictors found in formula.")
      return(NULL)
    }
    if (is.null(focal)) focal <- preds[1]
    
    need <- c(resp, preds)
    missing <- setdiff(need, names(data))
    if (length(missing) > 0) {
      log_warn("Skip std_beta_lm — missing vars: ", paste(missing, collapse = ", "))
      return(NULL)
    }
    
    z <- function(x) {
      if (!is.numeric(x)) return(x)
      s <- sd(x, na.rm = TRUE)
      if (is.na(s) || s == 0) return(x)
      (x - mean(x, na.rm = TRUE)) / s
    }
    df_use <- data[, need, drop = FALSE]
    df_std <- dplyr::mutate(df_use, dplyr::across(dplyr::everything(), z))
    
    mod <- lm(formula, data = df_std)
    smry <- summary(mod)
    cf   <- coef(smry)
    if (!(focal %in% rownames(cf))) {
      log_warn("std_beta_lm — focal term '", focal, "' not in model; available: ",
               paste(rownames(cf), collapse = ", "))
      return(NULL)
    }
    est <- cf[focal, "Estimate"]
    se  <- cf[focal, "Std. Error"]
    t   <- cf[focal, "t value"]
    p   <- cf[focal, "Pr(>|t|)"]
    ci  <- try(stats::confint(mod, level = 0.95), silent = TRUE)
    if (inherits(ci, "try-error")) {
      ci_lo <- NA_real_; ci_hi <- NA_real_
    } else {
      ci_lo <- ci[focal, 1]; ci_hi <- ci[focal, 2]
    }
    
    cat("\n", strrep("─", 81), "\n", sep = "")
    if (!is.null(note)) cat(note, "\n")
    cat(sprintf("%-20s %7s %7s %6s %6s     %s\n",
                "", "β", "S.E.", "t", "p", "[95% CI of β]"))
    cat(strrep("─", 81), "\n", sep = "")
    cat(sprintf("%-20s %7.3f (%0.3f) %6.3f  %5s     [%0.3f, %0.3f]\n",
                focal,
                est, se, t,
                ifelse(p < .001, "< .001", sprintf(".%03d", round(p, 3)*1000L)),
                ci_lo, ci_hi))
    cat(strrep("─", 81), "\n", sep = "")
    
    tibble::tibble(
      outcome  = resp,
      term     = focal,
      beta     = est,
      SE       = se,
      t        = t,
      p        = p,
      ci_lo    = ci_lo,
      ci_hi    = ci_hi,
      note     = note %||% ""
    )
  }
  
  need_delta <- !all(c("delta_attitude","delta_motivation","delta_se") %in% names(df))
  if (need_delta) {
    log_note("delta_* not found in df; computing from pre/post.")
    df <- df %>%
      dplyr::mutate(
        delta_attitude    = envAttitude_post     - envAttitude_pre,
        delta_motivation  = envMotivation_post   - envMotivation_pre,
        delta_se          = envSelfEfficacy_post - envSelfEfficacy_pre
      )
  }
  
  nudge_rt_cols <- grep("^nudge_(assi|inform)_rt_day\\d+$", names(df), value = TRUE)
  if (length(nudge_rt_cols) == 0) log_warn("No nudge RT columns matched; mean_nudge_rt will be NA.")
  df$mean_nudge_rt <- if (length(nudge_rt_cols) > 0) {
    rowMeans(log1p(df[, nudge_rt_cols, drop = FALSE]), na.rm = TRUE)
  } else NA_real_
  if ("nudgeType" %in% names(df)) {
    df$mean_nudge_rt[df$nudgeType == "struc"] <- 0
  }
  
  rt_will_cols  <- grep("^scenario\\d+_willingness_rt_day\\d+$", names(df), value = TRUE)
  rt_behav_cols <- grep("^scenario\\d+_behavior_rt_day\\d+$",    names(df), value = TRUE)
  if (length(rt_will_cols)  == 0) log_warn("No willingness RT columns matched.")
  if (length(rt_behav_cols) == 0) log_warn("No behaviour RT columns matched.")
  df$mean_beh_rt  <- if (length(rt_behav_cols) > 0) rowMeans(log1p(df[, rt_behav_cols, drop = FALSE]), na.rm = TRUE) else NA_real_
  df$mean_will_rt <- if (length(rt_will_cols)  > 0) rowMeans(log1p(df[, rt_will_cols,  drop = FALSE]), na.rm = TRUE) else NA_real_
  df <- df %>% dplyr::mutate(decision_rt = mean_beh_rt + mean_will_rt)
  
  log_step("Regressing internalisation/acceptance on nudge RT (standardized betas)")
  std_rows <- list()
  
  if (!"nudge_acceptance_post" %in% names(df)) {
    log_warn("nudge_acceptance_post is missing; acceptance regression will be skipped.")
  }
  df_ai <- if ("nudgeType" %in% names(df)) dplyr::filter(df, nudgeType %in% c("assis","infor")) else df
  
  r1 <- std_beta_lm(delta_attitude    ~ mean_nudge_rt, data = df,
                    focal = "mean_nudge_rt",
                    note = "Standardized coefficients — Δattitude ~ mean_nudge_rt")
  if (!is.null(r1)) std_rows <- append(std_rows, list(r1))
  
  r2 <- std_beta_lm(delta_motivation  ~ mean_nudge_rt, data = df,
                    focal = "mean_nudge_rt",
                    note = "Standardized coefficients — Δmotivation ~ mean_nudge_rt")
  if (!is.null(r2)) std_rows <- append(std_rows, list(r2))
  
  r3 <- std_beta_lm(delta_se          ~ mean_nudge_rt, data = df,
                    focal = "mean_nudge_rt",
                    note = "Standardized coefficients — Δself-efficacy ~ mean_nudge_rt")
  if (!is.null(r3)) std_rows <- append(std_rows, list(r3))
  
  if ("nudge_acceptance_post" %in% names(df)) {
    r4 <- std_beta_lm(nudge_acceptance_post ~ mean_nudge_rt, data = df,
                      focal = "mean_nudge_rt",
                      note = "Standardized coefficients — acceptance_post ~ mean_nudge_rt")
    if (!is.null(r4)) std_rows <- append(std_rows, list(r4))
  }
  
  std_table <- if (length(std_rows)) dplyr::bind_rows(std_rows) else NULL
  
  log_step("ANOVA: decision RT by nudgeType × nudgeIntensity")
  if (!all(c("decision_rt","nudgeType","nudgeIntensity","group") %in% names(df))) {
    log_warn("Skip decision RT ANOVA — columns missing.")
    fit_dt <- emm_dt_df <- NULL
  } else {
    df_noco <- df %>%
      dplyr::filter(group != "control") %>%
      dplyr::mutate(nudgeType = factor(nudgeType, levels = c("struc","assis","infor"))) %>%
      droplevels()
    car::leveneTest(decision_rt ~ nudgeType * nudgeIntensity, data = df_noco)
    fit_dt <- lm(decision_rt ~ nudgeType * nudgeIntensity, data = df_noco)
    print(car::Anova(fit_dt, type = 3))
    emm_dt <- emmeans::emmeans(fit_dt, ~ nudgeType | nudgeIntensity)
    emm_dt_contrasts <- summary(pairs(emm_dt), infer = TRUE, adjust = "BH")
    emm_dt_df <- as.data.frame(emm_dt_contrasts)
    print(emm_dt_df)
  }
  
  log_step("Building day-level RT and running PROCESS moderation")
  if (!all(c("ID","day") %in% names(df_day))) {
    log_warn("Skip PROCESS — df_day missing ID/day.")
    df_day2 <- NULL
    proc_out <- NULL
  } else {
    long_rt_w <- if (length(rt_will_cols) > 0) {
      df %>% dplyr::select(ID, dplyr::all_of(rt_will_cols)) %>%
        tidyr::pivot_longer(
          cols = dplyr::all_of(rt_will_cols),
          names_to = c("scenario","day"),
          names_pattern = "scenario(\\d+)_willingness_rt_day(\\d+)",
          values_to = "rt_will_ms"
        )
    } else tibble::tibble(ID = character(), scenario = character(), day = character(), rt_will_ms = numeric())
    
    long_rt_b <- if (length(rt_behav_cols) > 0) {
      df %>% dplyr::select(ID, dplyr::all_of(rt_behav_cols)) %>%
        tidyr::pivot_longer(
          cols = dplyr::all_of(rt_behav_cols),
          names_to = c("scenario","day"),
          names_pattern = "scenario(\\d+)_behavior_rt_day(\\d+)",
          values_to = "rt_behav_ms"
        )
    } else tibble::tibble(ID = character(), scenario = character(), day = character(), rt_behav_ms = numeric())
    
    long <- long_rt_w %>%
      dplyr::left_join(long_rt_b, by = c("ID","scenario","day")) %>%
      dplyr::mutate(
        rt_behav_ms = ifelse(rt_behav_ms < 200 | rt_behav_ms > 30000, NA, rt_behav_ms),
        rt_will_ms  = ifelse(rt_will_ms  < 200 | rt_will_ms  > 30000, NA, rt_will_ms),
        day = as.integer(day), scenario = as.integer(scenario)
      )
    df_day_rt <- long %>%
      dplyr::group_by(ID, day) %>%
      dplyr::summarise(
        mean_rt_will = mean(rt_will_ms,  na.rm = TRUE),
        mean_rt_be   = mean(rt_behav_ms, na.rm = TRUE),
        .groups = "drop"
      )
    df_day2 <- df_day %>%
      dplyr::left_join(df_day_rt, by = c("ID","day")) %>%
      dplyr::mutate(
        decision_rt = dplyr::if_else(
          rowSums(!is.na(dplyr::across(c(mean_rt_will, mean_rt_be)))) == 0,
          NA_real_,
          rowSums(dplyr::across(c(mean_rt_will, mean_rt_be)), na.rm = TRUE)
        ),
        decision_rt_log = log(decision_rt)
      )
    
    if (all(c("mean_behavior","mean_willingness","decision_rt_log") %in% names(df_day2))) {
      p <- NULL
      suppressMessages(suppressWarnings({
        utils::capture.output({
          p <- bruceR::PROCESS(
            df_day2,
            y        = "mean_behavior",
            x        = "mean_willingness",
            mods     = "decision_rt_log",
            clusters = c("day","ID"),
            ci       = "mcmc", nsim = 5000, seed = 1
          )
        })
      }))
      
      r1 <- p$results[[1]]
      mod_tbl <- tibble::as_tibble(r1$mod, .name_repair = "minimal")
      slopes_tbl <- r1$simple.slopes %>%
        tibble::as_tibble() %>%
        janitor::clean_names() %>%
        dplyr::rename(
          se    = dplyr::any_of(c("s_e", "se")),
          p     = dplyr::any_of(c("pval", "p")),
          ci_lo = dplyr::any_of(c("llci", "lower_ci", "lower", "ci_lower")),
          ci_hi = dplyr::any_of(c("ulci", "upper_ci", "upper", "ci_upper"))
        ) %>%
        dplyr::mutate(ci_label = sprintf("[%.3f, %.3f]", ci_lo, ci_hi)) %>%
        dplyr::select(decision_rt_log, effect, se, t, p, ci_lo, ci_hi, ci_label)
      conditional_tbl <- tibble::as_tibble(r1$conditional)
      
      rep_rule <- function(w = 70) paste0(strrep("─", w), "\n")
      fmt_p <- function(p) ifelse(p < .001, "< .001", sprintf("%.3f", p))
      p_stars <- function(p) ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", "")))
      pad <- function(x, w, align = "right") {
        if (is.na(x)) x <- ""
        if (align == "right") sprintf(paste0("%", w, "s"), x) else sprintf(paste0("%-", w, "s"), x)
      }
      fmt_coef <- function(est, se, p) paste0(sprintf("%0.3f", est), " ", p_stars(p), "\n", "(", sprintf("%0.3f", se), ")")
      
      mod1 <- lmerTest::lmer(mean_behavior ~ mean_willingness + (1|day) + (1|ID), data = df_day2)
      mod2 <- lmerTest::lmer(mean_behavior ~ mean_willingness * decision_rt_log + (1|day) + (1|ID), data = df_day2)
      
      cat("\n")
      cat("1. Regression\n")
      
      std_beta_tbl <- print_std_beta_table_hlmsummary(mod2, outcome_var = "mean_behavior")
      
      cat("\n")
      cat("2. Moderation\n\n")
      cat('Interaction Effect on "mean_behavior" (Y)\n')
      cat(rep_rule(61))
      cat(sprintf("%-35s %5s %4s %7s %6s\n", "", "F", "df1", "df2", "p"))
      cat(rep_rule(61))
      Fv  <- mod_tbl$F[1]; df1 <- mod_tbl$df1[1]; df2 <- mod_tbl$df2[1]; pv <- mod_tbl$pval[1]
      cat(sprintf("%-35s %5.2f %4.0f %7.0f %6s %s\n",
                  "mean_willingness * decision_rt_log", Fv, df1, df2, fmt_p(pv), p_stars(pv)))
      cat(rep_rule(61))
      
      cat('Simple Slopes: "mean_willingness" (X) ==> "mean_behavior" (Y)\n')
      cat(rep_rule(65))
      cat(sprintf(' %-17s %-8s %-8s %-6s %-8s %s\n',
                  '"decision_rt_log"', "Effect", "S.E.", "t", "p", "[95% CI]"))
      cat(rep_rule(65))
      ss <- slopes_tbl
      lab <- ss$decision_rt_log
      if (is.numeric(lab)) {
        lab <- c(sprintf("%.3f (- SD)", lab[1]),
                 sprintf("%.3f (Mean)", lab[2]),
                 sprintf("%.3f (+ SD)", lab[3]))
      }
      for (i in seq_len(nrow(ss))) {
        cat(sprintf(" %-17s %-8s %-8s %-6s %-8s [%0.3f, %0.3f]\n",
                    lab[i],
                    sprintf("%0.3f", ss$effect[i]),
                    sprintf("(%0.3f)", ss$se[i]),
                    sprintf("%0.3f", ss$t[i]),
                    paste0(fmt_p(ss$p[i]), " ", p_stars(ss$p[i])),
                    ss$ci_lo[i], ss$ci_hi[i]))
      }
      cat(rep_rule(65))
      
      proc_out <- list(
        interaction   = mod_tbl,
        simple_slopes = slopes_tbl,
        conditional   = conditional_tbl,
        model_y       = p$model.y,
        std_beta      = std_beta_tbl
      )
      
    } else {
      log_warn("Skip PROCESS — required cols missing in df_day2.")
      proc_out <- NULL
    }
  }  
  
  invisible(list(
    std_beta  = std_table,
    anova     = fit_dt,
    emm_pairs = emm_dt_df,
    df_day    = df_day2,
    df        = df,
    process   = proc_out
  ))
}


## ---------- C. MAIN PIPELINE ----------
log_info("Start analysis")

df <- read_data("../data/data.csv")
df <- prep_group_factors(df)
df <- prep_demographics(df)

vars_demo     <- c("Age","Sex","ethnicity","Nationality","occupation","income","education")
vars_men_pre  <- c("envAttitude_pre","envSelfEfficacy_pre","envMotivation_pre")
vars_baseline <- c(vars_demo, vars_men_pre)

attr_tbl <- attrition_summary(df)

log_info("Demographic table")
CreateTableOne(vars = vars_baseline, strata = "group", addOverall = TRUE, data = df)

df_day <- build_day_level(df, vars_baseline)

res_beh_d12 <- pairwise_vs_control_within_day(
  df_day, outcome = "mean_behavior", day_pair = c(1,2),
  caption = "Day 2 behaviour: group-vs-control (BH-adjusted)"
)

res_beh_within_d12 <- within_group_change(df_day, outcome = "mean_behavior", day_pair = c(1,2))

fit_beh  <- fit_exp_decay_vs_linear(df_day, outcome = "mean_behavior")

res_beh_within_d2d16  <- within_group_change(df_day, outcome = "mean_behavior",    day_pair = c(2,16))

res_event_low <- event_window_low(df_day)

res_will_d12 <- pairwise_vs_control_within_day(
  df_day, outcome = "mean_willingness", day_pair = c(1,2),
  caption = "Day 2 willingness: group-vs-control (BH-adjusted)"
)

res_will_within_d12 <- within_group_change(df_day, outcome = "mean_willingness", day_pair = c(1,2))
res_will_within_d2d16 <- within_group_change(df_day, outcome = "mean_willingness", day_pair = c(2,16))

fit_will <- fit_exp_decay_vs_linear(df_day, outcome = "mean_willingness")
lmm_will <- lmm_will_block(df_day)

gap_res   <- compute_gap_and_lmm(df_day);         df_day <- gap_res$df_day
intr_res  <- internalisation_block(df, vars_men_pre)
nf_res <- nudge_feedback_analyses(df,
                                  p_adjust = "BH",
                                  emmeans_spec = "intensity_by_type")
will_res  <- willingness_change_regressions(df)
rt_res    <- rt_block(df, df_day);                 df_day <- rt_res$df_day

log_done("All done.")
