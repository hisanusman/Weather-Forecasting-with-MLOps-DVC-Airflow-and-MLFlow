document.getElementById("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const humidity = document.getElementById("humidity").value;
    const windSpeed = document.getElementById("wind_speed").value;
    const condition = document.getElementById("condition").value;

    const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ humidity, wind_speed: windSpeed, condition_encoded: condition }),
    });

    const result = await response.json();
    document.getElementById("result").innerText = `Predicted Temperature: ${result.temperature_prediction}`;
});
