
special_tags = ['source_furry', 'source_anime', 'score_9', 'score_8_up', 'score_7_up', 'rating_explicit', 'rating_questionable', 'rating_safe']


for tagfile in Path(BASE_PATH).glob(r'**\*.txt'):
    with tagfile.open('+r') as f:
        file_tags = f.read().split(',')
        file_tags = tagfile.open('+r').read().split(',')
        new_tags = [tag.replace("_", " ") if tag not in special_tags else tag for tag in file_tags]
        new_tags = ','.join(new_tags)
        f.close()
    with tagfile.open('w') as f:
        f.write(new_tags)
        f.close()