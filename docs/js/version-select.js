window.addEventListener("DOMContentLoaded", function() {
  var REL_BASE_URL = "/OpenNMT";
  var ABS_BASE_URL = window.location.pathname;
  if (ABS_BASE_URL.substring(0,REL_BASE_URL.length)!==REL_BASE_URL) {
    // local deployment
    REL_BASE_URL = ""
  }

  var CURRENT_VERSION = ABS_BASE_URL.substring(REL_BASE_URL.length).split("/")[1];
  var v = CURRENT_VERSION.charCodeAt(0)
  if (isNaN(v) || v < 48 || v > 57) { CURRENT_VERSION = "master" }


  function makeSelect(options, selected) {
    var select = document.createElement("select");
    select.classList.add("md-header-nav__source");

    options.forEach(function(i) {
      var value = i
      if (value == "master") { value = "" }
      var option = new Option(i, value, undefined, i === selected);
      select.add(option);
    });

    return select;
  }

  var xhr = new XMLHttpRequest();
  xhr.open("GET", REL_BASE_URL + "/versions.json");
  xhr.onload = function() {
    var versions = JSON.parse(this.responseText);

    versions.push({"version": "master", "aliases":[]})

    var realVersion = versions.find(function(i) {
      return i.version === CURRENT_VERSION ||
             i.aliases.includes(CURRENT_VERSION);
    }).version;

    var versionPanel = document.createElement("div");
    versionPanel.id = "version-panel";
    versionPanel.innerHTML = "<label>Version: </label>";
    versionPanel.classList.add("md-flex__cell")
    versionPanel.classList.add("md-flex__cell--shrink");
    versionPanel.classList.add("md-header-nav__source");
    var select = makeSelect(versions.map(function(i) {
      return i.version;
    }), realVersion);
    versionPanel.appendChild(select);

    var left = document.querySelector("div.md-flex");
    left.appendChild(versionPanel);
    $('select').select2({ width: 'auto' }).on("change", function(event) {
      window.location.href = REL_BASE_URL + "/" + this.value;
    });
  };
  xhr.send();
});
