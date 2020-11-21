
let imgElement = document.createElement("IMG");
let inputElement = document.getElementById("fileInput");

inputElement.addEventListener("change",
    (e) => {
        imgElement.src = URL.createObjectURL(e.target.files[0]);
    },
    false);

imgElement.onload = function () {
    let original = cv.imread(imgElement);
    TurnToGrayScale(original);
    let standardSize = LimitSize(original);           
    cv.imshow('standardSize', standardSize);


    let scaleSpace = CreateScaleSpaceImage(standardSize);
    let dogSpace = CreateDOGSpaceImage(standardSize);

    cv.imshow('scaleSpace', scaleSpace);
    cv.imshow('dogSpace', dogSpace);
    //DrawKeypoints(original);           
};


function onOpenCvReady() {
    document.getElementById('status').innerHTML = 'OpenCV.js is ready.';
}