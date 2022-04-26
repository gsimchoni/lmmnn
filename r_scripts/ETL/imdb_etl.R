library(tidyverse)
library(tm)

# IMDB dataset from Kaggle: https://www.kaggle.com/datasets/wrandrall/imdb-new-dataset
imdb <- read_csv("imdb_db.csv")
colnames(imdb) <- janitor::make_clean_names(colnames(imdb))
imdb <- imdb %>% select(movie_name, movie_date, serie_date, movie_type,
                        number_of_votes, score, time_duration_min, director, description) %>%
  filter(is.na(serie_date), director != "[]") %>%
  select(-serie_date) %>%
  distinct(movie_name, .keep_all = TRUE)

# fix NAs
sum(is.na(imdb$movie_name))
imdb %>% count(movie_name, sort = T)
sum(is.na(imdb$movie_type))
imdb %>% count(movie_type, sort = T)
sum(is.na(imdb$movie_date))
imdb$movie_date[is.na(imdb$movie_date)] <- median(imdb$movie_date, na.rm = T)
imdb %>% count(movie_date, sort = T)
sum(is.na(imdb$number_of_votes))
summary(imdb$number_of_votes)
sum(is.na(imdb$score))
summary(imdb$score)
sum(is.na(imdb$time_duration_min))
imdb$time_duration_min[is.na(imdb$time_duration_min)] <- median(imdb$time_duration_min, na.rm = T)
summary(imdb$time_duration_min)
sum(is.na(imdb$director))
imdb %>% count(director, sort = T)
sum(is.na(imdb$description))

# recode director_id ID
imdb <- imdb %>% inner_join(
  imdb %>%
    distinct(director) %>%
    mutate(director_id = row_number() - 1),
  by = "director"
)

# recode type_id ID
imdb <- imdb %>% inner_join(
  imdb %>%
    distinct(movie_type) %>%
    mutate(type_id = row_number() - 1),
  by = "movie_type"
)

# text
corpus <- Corpus(VectorSource(imdb$description))
corpus <- tm_map(corpus, content_transformer(tolower))
removeHandles <- function(x) gsub("@[[:alnum:]]*", "", x)
corpus <- tm_map(corpus, content_transformer(removeHandles))
removeURL <- function(x) gsub("http[[:graph:]]+", "", x)
corpus <- tm_map(corpus, content_transformer(removeURL))
removeStrange <- function(x) gsub("(<.+>)+", "", x)
corpus <- tm_map(corpus, content_transformer(removeStrange))
myStopwords <- c(stopwords('english'),"economy","next","break","else","terms","while")
corpus <- tm_map(corpus, content_transformer(removeWords), myStopwords)
corpus <- tm_map(corpus, content_transformer(removePunctuation))
corpus <- tm_map(corpus, content_transformer(removeNumbers))
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
DTM <- DocumentTermMatrix(corpus, control=list(wordLengths=c(4, Inf)))
DTM <- removeSparseTerms(DTM, 0.99)

df_words <- as_tibble(as.matrix(DTM))
colnames(df_words) <- str_c("w_", colnames(df_words))

imdb <- bind_cols(imdb, df_words) %>%
  select(-c(movie_name, movie_type, director, description))
imdb %>%
  mutate(across(c(time_duration_min, number_of_votes, movie_date), ~(scale(.) %>% as.vector))) %>%
  write_csv("imdb_df2.csv")
