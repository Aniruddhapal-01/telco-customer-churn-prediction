
let churnChartInstance = null;
let contractChartInstance = null;
let tenureChartInstance = null;
let monthlyChartInstance = null;
let explanationChartInstance = null;

function renderDashboardCharts(data) {

    // Destroy previous charts before creating new ones
    [
        churnChartInstance,
        contractChartInstance,
        tenureChartInstance,
        monthlyChartInstance
    ].forEach(chart => {
        if (chart) chart.destroy();
    });

    // 1. Churn distribution
    churnChartInstance = new Chart(
        document.getElementById("churnChart"),
        {
            type: "doughnut",

            data: {
                labels: data.churn_distribution.labels,

                datasets: [{
                    data: data.churn_distribution.values,
                    backgroundColor: ["#635bdb", "#f59e0b"],
                    borderWidth: 0
                }]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        }
    );

    // 2. Contract-wise churn
    contractChartInstance = new Chart(
        document.getElementById("contractChart"),
        {
            type: "bar",

            data: {
                labels: data.contract_churn.labels,

                datasets: [
                    {
                        label: "Retained",
                        data: data.contract_churn.retained,
                        backgroundColor: "#635bdb"
                    },
                    {
                        label: "Churned",
                        data: data.contract_churn.churn,
                        backgroundColor: "#f59e0b"
                    }
                ]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                scales: {
                    x: {
                        stacked: true
                    },

                    y: {
                        stacked: true,
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: "Percentage"
                        }
                    }
                }
            }
        }
    );

    // 3. Tenure-wise churn
    tenureChartInstance = new Chart(
        document.getElementById("tenureChart"),
        {
            type: "bar",

            data: {
                labels: data.tenure_churn.labels,

                datasets: [{
                    label: "Churn Rate",
                    data: data.tenure_churn.values,
                    backgroundColor: "#635bdb",
                    borderRadius: 6
                }]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: "Churn Percentage"
                        }
                    }
                }
            }
        }
    );

    // 4. Average monthly charges
    monthlyChartInstance = new Chart(
        document.getElementById("monthlyChart"),
        {
            type: "bar",

            data: {
                labels: data.monthly_charges.labels,

                datasets: [{
                    label: "Average Monthly Charges",
                    data: data.monthly_charges.values,
                    backgroundColor: ["#635bdb", "#f59e0b"],
                    borderRadius: 6
                }]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: "Average Charges ($)"
                        }
                    }
                }
            }
        }
    );
}

function renderExplanationChart(data) {

    const canvas = document.getElementById("explanationChart");

    if (explanationChartInstance) {
        explanationChartInstance.destroy();
    }

    const features = data.features;

    const labels = features.map(item => item.feature);

    const values = features.map(item => item.impact);

    const colors = features.map(item =>
        item.impact > 0 ? "#f59e0b" : "#635bdb"
    );

    explanationChartInstance = new Chart(canvas, {

        type: "bar",

        data: {
            labels: labels,

            datasets: [{
                label: "SHAP Contribution",
                data: values,
                backgroundColor: colors,
                borderRadius: 5,
                barThickness: 20
            }]
        },

        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,

            plugins: {
                legend: {
                    display: false
                },

                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return " SHAP impact: " +
                                context.raw.toFixed(4);
                        }
                    }
                }
            },

            scales: {
                x: {
                    title: {
                        display: true,
                        text: "SHAP value (model output scale)"
                    },

                    grid: {
                        color: "#e8eaf2"
                    }
                },

                y: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

let confusionMatrixInstance = null;

function renderConfusionMatrix(data) {

    const cm = data.confusion_matrix;

    const canvas = document.getElementById(
        "confusionMatrixChart"
    );

    if (confusionMatrixInstance) {
        confusionMatrixInstance.destroy();
    }

    confusionMatrixInstance = new Chart(canvas, {

        type: "bar",

        data: {
            labels: [
                "Actual: Stayed",
                "Actual: Churned"
            ],

            datasets: [
                {
                    label: "Predicted: Stayed",
                    data: [
                        cm.true_negative,
                        cm.false_negative
                    ],
                    backgroundColor: "#635bdb"
                },
                {
                    label: "Predicted: Churned",
                    data: [
                        cm.false_positive,
                        cm.true_positive
                    ],
                    backgroundColor: "#f59e0b"
                }
            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            plugins: {
                legend: {
                    position: "top"
                }
            },

            scales: {
                x: {
                    stacked: true
                },

                y: {
                    stacked: true,
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: "Number of customers"
                    }
                }
            }
        }
    });
}
