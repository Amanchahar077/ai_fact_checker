const claimInput = document.getElementById("claim");
const submitButton = document.getElementById("submit");
const apiTestButton = document.getElementById("apiTest");
const resultEl = document.getElementById("result");
const verdictEl = document.getElementById("verdict");
const confidenceEl = document.getElementById("confidence");
const originEl = document.getElementById("origin");
const evidenceEl = document.getElementById("evidence");
const errorEl = document.getElementById("error");

// Map verdict strings to badge classes.
const verdictClass = (verdict) => {
  const normalized = (verdict || "").toLowerCase();
  if (normalized === "true" || normalized === "success") {
    return "success";
  }
  if (normalized === "false") {
    return "false";
  }
  return "warning";
};

// Render a success or error result in the UI.
const renderResult = (payload, isError = false) => {
  resultEl.classList.add("show");
  evidenceEl.innerHTML = "";
  errorEl.textContent = "";

  if (isError) {
    verdictEl.textContent = "Error";
    verdictEl.className = "badge false";
    confidenceEl.textContent = "";
    originEl.textContent = "";
    errorEl.textContent = payload;
    return;
  }

  const verdict = payload.verdict || "unverified";
  verdictEl.textContent = verdict;
  verdictEl.className = `badge ${verdictClass(verdict)}`;
  confidenceEl.textContent = `Confidence: ${Math.round((payload.confidence || 0) * 100)}%`;
  originEl.textContent = payload.api_used ? "Inference source: Hugging Face API" : "Inference source: local fallback";

  (payload.evidence || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    evidenceEl.appendChild(li);
  });
};

// Toggle loading state for the primary button.
const setLoading = (isLoading) => {
  submitButton.disabled = isLoading;
  submitButton.classList.toggle("loading", isLoading);
  submitButton.querySelector(".btn-text").textContent = isLoading ? "Checking" : "Check Fact";
};

// Submit claim for verification.
const postClaim = async () => {
  const claim = claimInput.value.trim();
  if (!claim) {
    renderResult("Enter a claim first.", true);
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/check_fact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || payload.error || "Fact check failed");
    }
    renderResult(payload);
  } catch (error) {
    renderResult(error.message, true);
  } finally {
    setLoading(false);
  }
};

// Run API connectivity test.
const testApi = async () => {
  apiTestButton.disabled = true;
  apiTestButton.textContent = "Testing...";

  try {
    const response = await fetch("/api_test");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "API test failed");
    }
    renderResult({
      verdict: payload.status,
      confidence: 1,
      api_used: true,
      evidence: [payload.message]
    });
  } catch (error) {
    renderResult(error.message, true);
  } finally {
    apiTestButton.disabled = false;
    apiTestButton.textContent = "Test API";
  }
};

submitButton.addEventListener("click", postClaim);
apiTestButton.addEventListener("click", testApi);

Array.from(document.querySelectorAll(".chip")).forEach((chip) => {
  chip.addEventListener("click", () => {
    claimInput.value = chip.dataset.claim || "";
    claimInput.focus();
  });
});
