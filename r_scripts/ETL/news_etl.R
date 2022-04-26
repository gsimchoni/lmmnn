library(tidyverse)
library(tm)

# The News popularity in social media dataset from UCI ML repository: 
# https://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms
news <- read_csv("News_Final.csv") %>%
  distinct(IDLink, .keep_all = T) %>%
  mutate(hour = lubridate::hour(PublishDate),
         day = lubridate::wday(PublishDate),
         month = lubridate::month(PublishDate)) %>%
  filter(Facebook != -1)

# recode title ID
news <- news %>% inner_join(
  news %>%
    distinct(Title) %>%
    mutate(title_id = row_number() - 1),
  by = "Title"
)

# recode source ID
news <- news %>% inner_join(
  news %>%
    distinct(Source) %>%
    mutate(source_id = row_number() - 1),
  by = "Source"
)

corpus <- Corpus(VectorSource(news$Headline))
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

df <- bind_cols(news, df_words)

df_topic <- as_tibble(model.matrix(~ 0 + df$Topic))
colnames(df_topic) <- c("topic_economy", "topic_microsoft", "topic_obama", "topic_palestine")

df %>%
  bind_cols(df_topic) %>%
  write_csv("news_df2.csv")
