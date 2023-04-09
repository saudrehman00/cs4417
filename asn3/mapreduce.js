function myMapper() {
    this.entities.hashtags.forEach(function (hashtag){
        emit(hashtag.text, 1);
    })
}

function myReducer(key, values) {
    return Array.sum(values);
}

db.tweets.mapReduce(myMapper, myReducer, { out: "mroutput" });

db.mroutput.find().sort({ value: -1 });
