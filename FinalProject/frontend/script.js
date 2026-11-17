
const churnForm = document.getElementById("churnForm");

const resultCard = document.getElementById("resultCard");
const analyticsSection = document.getElementById("analyticsSection");
const explanationSection = document.getElementById("explanationSection");
const performanceSection = document.getElementById("performanceSection");

const resultTitle = document.getElementById("resultTitle");
const resultMessage = document.getElementById("resultMessage");
const resultProbability = document.getElementById("resultProbability");
const resultRisk = document.getElementById("resultRisk");
const resultIcon = document.getElementById("resultIcon");

const predictButton = document.querySelector(".predict-btn");


// ------------------------------------
// 1. Reusable API response handler
// ------------------------------------

async function fetchJSON(url, options = {}) {

    const response = await fetch(url, options);

    // Read response only once
    const responseText = await response.text();

    console.log(`API: ${url}`);
    console.log("HTTP Status:", response.status);
    console.log("Raw Response:", responseText);

    // Check for empty response
    if (!responseText.trim()) {
        throw new Error(
            `${url} returned an empty response (HTTP ${response.status}).`
        );
    }

    let data;

    // Safely parse JSON
    try {
        data = JSON.parse(responseText);
    } catch (error) {
        console.error("Invalid JSON response:", responseText);

        throw new Error(
            `${url} returned invalid JSON. Check the Flask backend and route.`
        );
    }

    // Check HTTP status
    if (!response.ok) {
        throw new Error(
            data.error || `Request failed with HTTP ${response.status}`
        );
    }

    return data;
}


// ------------------------------------
// 2. Prediction form
// ------------------------------------

churnForm.addEventListener("submit", async function (event) {

    event.preventDefault();

    if (!churnForm.checkValidity()) {
        churnForm.reportValidity();
        return;
    }

    const formData = new FormData(churnForm);
    const customerData = Object.fromEntries(formData.entries());

    // Convert numerical values
    customerData.SeniorCitizen = Number(customerData.SeniorCitizen);
    customerData.tenure = Number(customerData.tenure);
    customerData.MonthlyCharges = Number(customerData.MonthlyCharges);
    customerData.TotalCharges = Number(customerData.TotalCharges);

    // Loading state
    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";

    resultCard.hidden = true;
    analyticsSection.hidden = true;
    explanationSection.hidden = true;
    performanceSection.hidden = true;


    try {

        // ------------------------------------
        // 3. Customer prediction
        // ------------------------------------

        const data = await fetchJSON("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(customerData)

        });


        // Display prediction
        resultCard.hidden = false;

        resultTitle.textContent =
            data.prediction === "Yes"
                ? "Customer Likely to Churn"
                : "Customer Likely to Stay";

        resultMessage.textContent =
            "Prediction generated successfully.";

        resultProbability.textContent =
            "Churn Probability: " + data.churn_probability + "%";

        resultRisk.textContent =
            "Risk Level: " + data.risk_level;

        resultIcon.textContent =
            data.prediction === "Yes" ? "!" : "✓";


        // ------------------------------------
        // 4. Overall dataset analytics
        // ------------------------------------

        try {

            const analyticsData = await fetchJSON("/analytics");

            analyticsSection.hidden = false;

            renderDashboardCharts(analyticsData);

        } catch (analyticsError) {

            console.error("Analytics error:", analyticsError);

            analyticsSection.hidden = true;

        }


        // ------------------------------------
        // 5. Individual SHAP explanation
        // ------------------------------------

        try {

            const explanationData = await fetchJSON("/explain", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(customerData)

            });

            explanationSection.hidden = false;

            renderExplanationChart(explanationData);

        } catch (explanationError) {

            explanationSection.hidden = true;

            console.error(
                "SHAP explanation error:",
                explanationError
            );

        }


        // ------------------------------------
        // 6. Model performance evaluation
        // ------------------------------------

        try {

            const performanceData = await fetchJSON(
                "/model-performance"
            );

            document.getElementById("accuracyValue").textContent =
                performanceData.accuracy + "%";

            document.getElementById("precisionValue").textContent =
                performanceData.precision + "%";

            document.getElementById("recallValue").textContent =
                performanceData.recall + "%";

            document.getElementById("f1Value").textContent =
                performanceData.f1_score + "%";

            document.getElementById("evaluationInfo").textContent =
                "Test samples: " + performanceData.test_samples +
                " | Classification threshold: " +
                performanceData.threshold;

            performanceSection.hidden = false;

            renderConfusionMatrix(performanceData);

        } catch (performanceError) {

            performanceSection.hidden = true;

            console.error(
                "Model evaluation error:",
                performanceError
            );

        }


        // Scroll to results
        resultCard.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });


    } catch (error) {

        resultCard.hidden = false;

        resultTitle.textContent = "Prediction Error";
        resultMessage.textContent = error.message;
        resultProbability.textContent = "";
        resultRisk.textContent = "";
        resultIcon.textContent = "!";

        console.error("Prediction error:", error);

    } finally {

        predictButton.disabled = false;

        predictButton.innerHTML =
            'Predict Churn <span>→</span>';

    }

});


// ------------------------------------
// 7. Reset form
// ------------------------------------

churnForm.addEventListener("reset", function () {

    resultCard.hidden = true;
    analyticsSection.hidden = true;
    explanationSection.hidden = true;
    performanceSection.hidden = true;

});
