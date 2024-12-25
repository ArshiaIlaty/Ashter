from bing_image_downloader import downloader

query_string = "your_query_string"  # Replace "your_query_string" with the actual query string

downloader.download(query_string, limit=100,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60)