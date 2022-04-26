library(tidyverse)
library(lubridate)
library(janitor)
library(scales)

# US Adult Asthma dataset on cencus tract level from CDC: https://www.cdc.gov/nceh/tracking/topics/asthma.htm
# US demographic features come from US Census Demographic dataset from Kaggle: https://www.kaggle.com/datasets/muonneutrino/us-census-demographic-data

# after downloading all 52 states zip files from CDC
# setwd("asthma/")
# 
# for (f in list.files("asthma")) {
#   unzip(f)
# }

# after unzipping all of them in the asthma folder
df <- list.files(path = "asthma", pattern = "*.csv$") %>%
  map(~read_csv(str_c("asthma/", .x))) %>%
  reduce(rbind) 

colnames(df) <- make_clean_names(colnames(df))

df <- df %>%
  select(census_tract, value) %>%
  filter(value != "Low Population") %>%
  mutate(value = parse_number(value))

# US census ETL
df_census <- read_csv("us_cencus/acs2017_census_tract_data.csv")
df_county <- read_csv("us_cencus/acs2017_county_data.csv")
df_county_lat_lon <- read_csv("us_cencus/us_county.csv")

df_county <- df_county %>% inner_join(df_county_lat_lon, by = c("CountyId" = "fips")) %>%
  select(CountyId, State, County, lat, long)
df_census <- df_census %>% inner_join(df_county)
colnames(df_census) <- make_clean_names(colnames(df_census))

# recode pop, income
df_census <- df_census %>%
  mutate(men = men/total_pop,
         women = women / total_pop,
         employed = employed / total_pop,
         voting_age_citizen = voting_age_citizen / total_pop,
         total_pop = log(total_pop),
         income = log(income)) %>%
  select(-income_err, -income_per_cap, -income_per_cap_err, -state, -county, -county_id)

# recode percentages
percent_cols <- c("Hispanic", "White", "Black", "Native", "Asian", "Pacific",
                  "Poverty", "ChildPoverty", "Professional", "Service", "Office",
                  "Construction", "Production", "Drive", "Carpool", "Transit", "Walk",
                  "OtherTransp", "WorkAtHome", "MeanCommute", "PrivateWork", "PublicWork",
                  "SelfEmployed", "FamilyWork", "Unemployment")

df_census <- df_census %>% mutate(across(all_of(make_clean_names(percent_cols)), ~.x/100))


# join with asthma data
df <- df %>% mutate(census_tract = as.numeric(census_tract)) %>%
  inner_join(df_census, by = c("census_tract" = "tract_id")) %>%
  select(-census_tract, asthma = value)

# recode location
location_df <- df %>%
  select(lat, long) %>%
  arrange(lat, long) %>%
  distinct() %>%
  filter(
    between(lat, 20, 50),
    between(long, -150, -50)
  ) %>%
  mutate(location_id = 0 : (n() - 1))

df <- df %>% inner_join(location_df)

df$lat <- rescale(df$lat, to = c(-10, 10))
df$long <- rescale(df$long, to = c(-10, 10))

# no NAs
df <- df %>% drop_na()

df %>% write_csv("asthma_df.csv")
