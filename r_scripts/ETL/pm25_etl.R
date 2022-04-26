library(tidyverse)
library(lubridate)
library(janitor)
library(scales)

# US Air Quality data on census tract level from CDC: 
# https://data.cdc.gov/Environmental-Health-Toxicology/Daily-Census-Tract-Level-PM2-5-Concentrations-2016/7vu4-ngxx
# US demographic features come from US Census Demographic dataset from Kaggle: https://www.kaggle.com/datasets/muonneutrino/us-census-demographic-data

df <- read_csv("Daily_Census_Tract-Level_PM2.5_Concentrations__2016.csv") %>%
  filter(date == "01JAN2016")

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
  select(-income_err, -income_per_cap, -income_per_cap_err, -state, -county)

# recode percentages
percent_cols <- c("Hispanic", "White", "Black", "Native", "Asian", "Pacific",
                  "Poverty", "ChildPoverty", "Professional", "Service", "Office",
                  "Construction", "Production", "Drive", "Carpool", "Transit", "Walk",
                  "OtherTransp", "WorkAtHome", "MeanCommute", "PrivateWork", "PublicWork",
                  "SelfEmployed", "FamilyWork", "Unemployment")

df_census <- df_census %>% mutate(across(all_of(make_clean_names(percent_cols)), ~.x/100))


# join with PM2.5 data
df <- df %>% inner_join(df_census, by = c("ctfips" = "tract_id")) %>%
  select(-year, -date, -statefips, -countyfips, -ctfips, -DS_PM_stdd,
         -latitude, -longitude, pm25 = DS_PM_pred)

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

df %>% write_csv("pm25_df.csv")
