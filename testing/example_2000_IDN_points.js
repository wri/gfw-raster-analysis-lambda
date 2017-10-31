var AWS = require('aws-sdk');
const uuidv4 = require('uuid/v4');

AWS.config.update({region: 'us-east-1'});

var lambda = new AWS.Lambda();

var fs = require('fs');
var geojson = JSON.parse(fs.readFileSync(__dirname + '/' + 'idn_points.geojson', 'utf8'));

var features = geojson.features.slice(0,1)

for (var i = 0; i < features.length; i++) {

  var lon = features[i].geometry.coordinates[0]
  var lat = features[i].geometry.coordinates[1]

  var out_dir = lat + '_' + lon + '_' + uuidv4()

  var payload = {
    "queryStringParameters": {
        "out_dir": out_dir,
        "lat": lat,
        "lon": lon
        }
    }

  var params = {
    FunctionName: 'palm-risk-poc-dev-mill',
    Payload: JSON.stringify(payload)
  }

  lambda.invoke(params, function(err, data) {
    if (err) console.log(err, err.stack); // an error occurred
  });

}
