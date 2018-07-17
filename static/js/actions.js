document.getElementById("submitButton").onclick = function(){
    const canvas = document.getElementById('myCanvas');
    const answerDiv = document.getElementById("answer");
    fetch('/recognize', {
        body: canvas.toDataURL(),
        method: 'POST',
    })
    .then(response => {
        if (response.ok) {
          return Promise.resolve(response);
        }
        else {
          return Promise.reject(new Error('Failed to load'));
        }
    })
    .then(response => response.json())
    .then(data => {
      answerDiv.innerHTML = data.prediction;
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    })
    .catch(function(error) {
        answerDiv.innerHTML = error.message;
    });
};
