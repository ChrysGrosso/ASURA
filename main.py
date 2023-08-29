import steamreviews_verify_false #pour pouvoir bénéficier de verify=False


#Script to retrieve the Json with conditions
request_params = dict()
request_params['language'] = 'english'
request_params['filter'] = 'recent' #recent for reviews written within day_range ; updated for reviews updated within day_range
request_params['day_range'] = '28' # 28 days back from current date
review_dict, query_count = steamreviews_verify_false.download_reviews_for_app_id(1245620, chosen_request_params=request_params)
