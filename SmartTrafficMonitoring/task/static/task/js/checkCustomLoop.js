const defaultRadio = document.querySelector('input[value="default"]');
const customRadio = document.querySelector('input[value="custom"]');
const customFileInput = document.getElementById("custom-file");

customRadio.addEventListener("change", function() {
    if (this.checked) {
      customFileInput.style.display = "block";
      customFileInput.required = true;
    } else {
      customFileInput.style.display = "none";
      customFileInput.required = false;
    }
});

defaultRadio.addEventListener("change", function() {
if (this.checked) {
    customFileInput.style.display = "none";
    customFileInput.required = false;
  } else {
    customFileInput.style.display = "block";
    customFileInput.required = true;
  }
});