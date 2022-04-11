library(tidyverse)

spotify_songs <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv')
subgenre_df <- spotify_songs %>% group_by(track_id) %>% summarize(pl_subgenres = list(playlist_subgenre)) %>% mutate(pl_subgenres = map_chr(pl_subgenres, function(gs) str_c(sort(gs), collapse = ",")))
playlist_df <- spotify_songs %>% group_by(track_id) %>% summarize(pls = list(playlist_id)) %>% mutate(playlist_ids = map_chr(pls, function(gs) str_c(sort(gs), collapse = ","))) %>% select(-pls)

spotify_df <- spotify_songs %>%
  select(track_id, track_artist, track_album_id, track_album_release_date, 12:23) %>%
  distinct(track_id, .keep_all = TRUE) %>%
  mutate(
    year = lubridate::year(as.Date(track_album_release_date)),
    month = lubridate::month(as.Date(track_album_release_date)),
    wday = lubridate::wday(as.Date(track_album_release_date))
  ) %>%
  inner_join(subgenre_df) %>%
  inner_join(playlist_df)

# fix NAs
spotify_df$year[is.na(spotify_df$year)] <- as.numeric(spotify_df$track_album_release_date[is.na(spotify_df$year)])
spotify_df$year[is.na(spotify_df$year)] <- median(spotify_df$year, na.rm = T)
spotify_df$month[is.na(spotify_df$month)] <- median(spotify_df$month, na.rm = T)
spotify_df$wday[is.na(spotify_df$wday)] <- median(spotify_df$wday, na.rm = T)

# recode artist ID
spotify_df <- spotify_df %>% inner_join(
  spotify_df %>%
    distinct(track_artist) %>%
    mutate(artist_id = row_number() - 1),
  by = "track_artist"
)

# recode album_id ID
spotify_df <- spotify_df %>% inner_join(
  spotify_df %>%
    distinct(track_album_id) %>%
    mutate(album_id = row_number() - 1),
  by = "track_album_id"
)

# recode playlist_id ID
spotify_df <- spotify_df %>% inner_join(
  spotify_df %>%
    distinct(playlist_ids) %>%
    mutate(playlist_id = row_number() - 1),
  by = "playlist_ids"
)

# recode subgenre_id ID
spotify_df <- spotify_df %>% inner_join(
  spotify_df %>%
    distinct(pl_subgenres) %>%
    mutate(subgenre_id = row_number() - 1),
  by = "pl_subgenres"
)

spotify_df %>%
  mutate(across(c(duration_ms, loudness, tempo, year, month, wday, key), ~(scale(.) %>% as.vector))) %>%
  write_csv("spotify_df.csv")
