function predictDiet() {
    // Collect input values from the form
    var age = document.getElementById('age').value;
    var gender = document.getElementById('gender').value;
    var bmi = document.getElementById('bmi').value;
    var genetic = document.getElementById('genetic').value;

    // Prepare the data to be sent to the server
    var formData = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "genetic": genetic
    };

    // Use fetch API to send the data to the server and get the prediction
    fetch('/predict_diet', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result
        document.getElementById('recommendation').innerHTML = 'Diet Recommendation: ' + data['diet_recommendation'];
    })
     
    .catch((error) => {
        console.error('Error:', error);
    });
}
function predictLoneliness() {
    var age = document.getElementById('age').value;
    var gender = document.getElementById('gender').value;
    var maritalStatus = document.getElementById('maritalStatus').value;
    var livingSituation = document.getElementById('livingSituation').value;
    var socialNetworkSize = document.getElementById('socialNetworkSize').value;
    var socialParticipation = document.getElementById('socialParticipation').value;

    var formData = {
        "age": age,
        "gender": gender,
        "maritalStatus": maritalStatus,
        "livingSituation": livingSituation,
        "socialNetworkSize": socialNetworkSize,
        "socialParticipation": socialParticipation
    };

    fetch('/predict_loneliness', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loneliness').innerHTML = 'Loneliness Assessment: ' + data['loneliness_assessment'];
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
function predictExerciseRecommendation() {
    var age = document.getElementById('age').value;
    var gender = document.getElementById('gender').value;
    var bmi = document.getElementById('bmi').value;
    var healthConditions = document.getElementById('healthConditions').value;
    var mobility = document.getElementById('mobility').value;
    

    var formData = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "healthConditions": healthConditions,
        "mobility": mobility
        
    };

    fetch('/predict_exercise', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('exerciseRecommendation').innerHTML = 'Exercise Recommendation: ' + data['exercise_recommendation'];

    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
function predictGeneralHealthStatus() {
    // Collect form data
    var age = document.getElementById('age').value;
    var gender = document.getElementById('gender').value;
    var bmi = document.getElementById('bmi').value;
    var healthConditions = document.getElementById('healthConditions').value;

    // Prepare the data payload
    var formData = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "existingHealthConditions": healthConditions
    };

    // Send the data to the Flask backend using fetch()
    fetch('/predict_general_health_status', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())  // Parse the JSON response
    .then(data => {
        // Display the prediction result
        document.getElementById('generalHealthStatus').innerHTML = 'General Health Status: ' + data['general_health_status'];
    })
    .catch((error) => {
        // Handle any errors
        console.error('Error:', error);
    });
}
