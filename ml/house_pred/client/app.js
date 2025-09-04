function getBathValue() {
  const uiBathrooms = document.getElementsByName("uiBathrooms");
  for (let i = 0; i < uiBathrooms.length; i++) {
    if (uiBathrooms[i].checked) {
      return parseInt(uiBathrooms[i].value);
    }
  }
  return -1; // Invalid Value
}

function getBHKValue() {
  const uiBHK = document.getElementsByName("uiBHK");
  for (let i = 0; i < uiBHK.length; i++) {
    if (uiBHK[i].checked) {
      return parseInt(uiBHK[i].value);
    }
  }
  return -1; // Invalid Value
}

function onClickedEstimatePrice() {
  console.log("Estimate price button clicked");
  const sqft = document.getElementById("uiSqft").value;
  const bhk = getBHKValue();
  const bathrooms = getBathValue();
  const location = document.getElementById("uiLocations").value;
  const estPrice = document.getElementById("uiEstimatedPrice");

  const url = "http://127.0.0.1:5000/predict_home_price";

  $.post(url, {
    total_sqft: parseFloat(sqft),
    bhk: bhk,
    bath: bathrooms,
    location: location
  }, function (data, status) {
    if (data.estimated_price !== undefined) {
      estPrice.innerHTML = `<h2>${data.estimated_price} Lakh</h2>`;
    } else {
      estPrice.innerHTML = `<h2>Error: Unable to estimate price</h2>`;
    }
    console.log(status);
  });
}

function onPageLoad() {
  console.log("document loaded");
  const url = "http://127.0.0.1:5000/get_location_names";
  $.get(url, function (data, status) {
    console.log("got response for get_location_names request");
    if (data && data.locations) {
      const uiLocations = document.getElementById("uiLocations");
      uiLocations.innerHTML = "";
      data.locations.forEach(function (loc) {
        const opt = new Option(loc, loc);
        uiLocations.appendChild(opt);
      });
    }
  });
}

window.onload = onPageLoad;