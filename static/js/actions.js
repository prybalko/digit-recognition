document.getElementById("submitButton").onclick = function(){
    let canvas = document.getElementById('myCanvas');
    let dataURL = canvas.toDataURL();
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
    .then(response => response.json()) // parse response as JSON
    .then(data => {
      const context = canvas.getContext('2d');
      console.log(data.prediction);
      context.clearRect(0, 0, canvas.width, canvas.height);
    })
    .catch(function(error) {
        console.log(`Error: ${error.message}`);
    });
};
