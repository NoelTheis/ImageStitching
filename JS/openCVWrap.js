let padX;
let padY;
let maxX = 100;
let maxY = 100;
let gaussSequenceLength = 6;
let downSampleCount = 4;
let scaleSpace;
let dogSpace;


//Color stuff
function TurnToGrayScale(image) {
    cv.cvtColor(image, image, cv.COLOR_BGR2GRAY);
}

//Size stuff
function LimitSize(image)  { 
    if(image.cols > maxX || image.rows > maxY){
        let xFactor = maxX / image.cols;
        let yFactor = maxY / image.rows;
        let smallestFactor = Math.min(xFactor, yFactor);
        let dest = new cv.Mat();
        cv.resize(image, dest, new cv.Size(), smallestFactor, smallestFactor, cv.INTER_NEAREST);
        return dest;
    }
    return image;      
}

function DownSample(image){
    let dest = new cv.Mat();
    cv.resize(image, dest, new cv.Size(), 0.5, 0.5, cv.INTER_NEAREST);
    return dest;
}

function UpSample(image){
    let dest = new cv.Mat();
    cv.resize(image, dest, new cv.Size(), 2, 2, cv.INTER_LINEAR);
    return dest;
}

function PadImage(image) {
    let top = 0;
    let left = 0;
    let right = 0;
    let bottom = 0;
    let missingWidth = padX - image.cols;
    let missingHeight = padY - image.rows;

    if(missingWidth > 0){
        left = Math.floor(missingWidth / 2.0);
        right = Math.ceil(missingWidth / 2.0);
    }
    if(missingHeight > 0){
        top = Math.floor(missingHeight / 2.0);
        bottom = Math.ceil(missingHeight / 2.0);
    }

    let dest = new cv.Mat();
    cv.copyMakeBorder(image, dest, top, bottom, left, right, cv.BORDER_CONSTANT, new cv.Scalar(255,255,255,255));
    //cv.copyMakeBorder(image, dest, top, bottom, left, right, cv.BORDER_TRANSPARENT);
    return dest;
}

//Combining images
function CombineMatVectorsToImage(rows){
    let rowVector = new cv.MatVector();
    rows.forEach(row => {
        let rowImage = myHConcat(row);
        rowVector.push_back(rowImage);       
    });
    let scaleSpaceImage = myVConcat(rowVector);
    return scaleSpaceImage;
}

function myHConcat(vector){
    let image = new cv.Mat();
    cv.hconcat(vector, image);
    return image;
}

function myVConcat(vector){
    let image = new cv.Mat();
    cv.vconcat(vector, image);
    return image;
}

//Gauss stuff
function CreateScaleSpaceImage(image) {
    scaleSpace = CreateScaleSpace(image);

    return CombineMatVectorsToImage(scaleSpace);
}

function CreateScaleSpace(image){
    let rows = [];
    let currentStartImage = UpSample(image);
    padX = currentStartImage.cols;
    padY = currentStartImage.rows;
    //currentStartImage = PadImage(currentStartImage);
    for (let i = 0; i < downSampleCount; i++) {
        let row = CreateGaussSequence(currentStartImage);
        rows.push(row);
        let downSampled = DownSample(row.get(gaussSequenceLength - 1));
        currentStartImage = PadImage(downSampled);
    }
    return rows; 
}

function CreateGaussSequence(image){
    let matVec = new cv.MatVector();
    let currentImage = image;
    for (let i = 0; i < gaussSequenceLength; i++) {
        currentImage = ApplyGaussian(currentImage, 0);
        matVec.push_back(currentImage);
    } 
    return matVec;
}

function ApplyGaussian(image, deviation){
    let dest = new cv.Mat();
    cv.GaussianBlur(image, dest, new cv.Size(5,5),deviation,deviation, cv.BORDER_REPLICATE);
    return dest;
}

//Difference of gaussian stuff
function CreateDOGSpaceImage(image) {
    scaleSpace = CreateScaleSpace(image);
    dogSpace = CreateDOGSpace(scaleSpace);

    return CombineMatVectorsToImage(dogSpace);
}

function CreateDOGSpace(scaleSpace){
    let dogSpace = [];
    scaleSpace.forEach(scaleSpaceRow => {
        let dogRow = CreateDOGRow(scaleSpaceRow);
        dogSpace.push(dogRow);
    });
    return dogSpace;
}

function CreateDOGRow(scaleSpaceRow){
    let dogRow = new cv.MatVector();
    for(let i = 0;i < gaussSequenceLength - 1; i++){
        let dog = new cv.Mat();
        let mask = new cv.Mat();
        let dtype = -1;
        cv.subtract(scaleSpaceRow.get(i), scaleSpaceRow.get(i+1), dog, mask, dtype);
        //cv.absdiff(scaleSpaceRow.get(i), scaleSpaceRow.get(i+1), dog);
        //dog = CreateDOG(scaleSpaceRow.get(i), scaleSpaceRow.get(i+1));
        dogRow.push_back(dog);
    }
    return dogRow;
}

// function CreateDOG(gauss1, gauss2){
//     let diffImage = new cv.Mat();
//     cv.absdiff(gauss1, gauss2, diffImage);

//     let foregroundMask = new cv.Mat.zeros(diffImage.rows, diffImage.cols, cv.CV_8U);

//     let threshold = 30.0;
//     let dist;

//     for(let j=0; j<diffImage.rows; ++j){
//         for(let i=0; i<diffImage.cols; ++i)
//         {
//             let pix = diffImage.ucharAt(j,i);

//             dist = (pix[0]*pix[0] + pix[1]*pix[1] + pix[2]*pix[2]);
//             dist = Math.sqrt(dist);

//             if(dist>threshold)
//             {
//                 foregroundMask.ucharAt(j,i) = 255;
//             }
//         }
//     }

// }

//Keypoint stuff

function DrawKeypoints(image) {
    let orb = new cv.ORB(100);
    let kp = new cv.KeyPointVector();
    orb.detect(image, kp);
    cv.drawKeypoints(image,kp,image);
}
