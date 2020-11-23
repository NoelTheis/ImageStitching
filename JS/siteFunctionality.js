
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
    let standardSizeImage = LimitSize(original);           
    cv.imshow('standardSize', standardSizeImage);
    let upscaledImage = UpSample(standardSizeImage);
    cv.imshow('upscaled', upscaledImage);

    let scaleSpaceImage = CreateScaleSpaceImage(upscaledImage);
    let dogSpaceImage = CreateDoGsImage();

    cv.imshow('scaleSpace', scaleSpaceImage);
    cv.imshow('dogSpace', dogSpaceImage);
    //DrawKeypoints(original);           
};


function onOpenCvReady() {
    document.getElementById('status').innerHTML = 'OpenCV.js is ready.';
}