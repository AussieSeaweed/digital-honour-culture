def pull_tweets(client, query, iter_count, max_results=100):
    next_token = None

    for _ in range(iter_count):
        response = client.search_recent_tweets(
            query,
            expansions=['author_id'],
            max_results=max_results,
            next_token=next_token,
        )
        tweets = response[0]
        next_token = response[3]['next_token']

        yield from tweets
