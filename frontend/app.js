document.getElementById("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const humidity = parseFloat(document.getElementById("humidity").value);
    const windSpeed = parseFloat(document.getElementById("wind_speed").value);
    const condition = parseFloat(document.getElementById("condition").value);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                humidity,
                wind_speed: windSpeed,
                condition_encoded: condition,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        document.getElementById("result").innerText = `Predicted Temperature: ${result.predicted_temperature}`;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Failed to get prediction. Check console for details.";
    }
});
