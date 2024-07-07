const DROP_AREA = document.getElementById("drag-n-drop-area");
const INPUT_IMAGE = document.getElementById("inputImage");
const IMAGE_VIEW = document.getElementById("img-view");

INPUT_IMAGE.addEventListener("change", uploadImage);

function uploadImage() {
  let imgLink = URL.createObjectURL(INPUT_IMAGE.files[0]);
  IMAGE_VIEW.style.backgroundImage = `url(${imgLink})`;
  IMAGE_VIEW.style.border = 0;
  IMAGE_VIEW.textContent = "";
}

DROP_AREA.addEventListener("dragover", (e) => e.preventDefault());
DROP_AREA.addEventListener("drop", (e) => {
  e.preventDefault();
  INPUT_IMAGE.files = e.dataTransfer.files;
  uploadImage();
});

if (INPUT_IMAGE.files.length > 0) {
  uploadImage();
}
