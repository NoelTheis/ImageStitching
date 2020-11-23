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

    return dest;
}

//Combining images
function CombineMatVectorArrayToImage(rows){
    let rowImages = new cv.MatVector();
    padX = rows[0].get(0).cols;
    padY = rows[0].get(0).rows;
    rows.forEach(row => {
        let paddedRow = new cv.MatVector();      
        for(let i = 0; i < row.size(); i++){
            let padded = PadImage(row.get(i));
            paddedRow.push_back(padded);
        }
        let rowImage = myHConcat(paddedRow);
        rowImages.push_back(rowImage);       
    });
    let finalImage = myVConcat(rowImages);
    return finalImage;
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

    return CombineMatVectorArrayToImage(scaleSpace);
}

function CreateScaleSpace(image){
    let rows = [];
    let firstImageInRow = image;
    for (let i = 0; i < downSampleCount; i++) {
        let row = CreateScaleSpaceRow(firstImageInRow);
        rows.push(row);
        firstImageInRow = DownSample(row.get(row.size() - 1));
    }
    return rows; 
}

function CreateScaleSpaceRow(image){
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
function CreateDoGsImage() {
    dogSpace = CreateDoGRows(scaleSpace);

    return CombineMatVectorArrayToImage(dogSpace);
}

function CreateDoGRows(scaleSpace){
    let DoGRows = [];
    scaleSpace.forEach(scaleSpaceRow => {
        let DoGRow = CreateDoGRow(scaleSpaceRow);
        DoGRows.push(DoGRow);
    });
    return DoGRows;
}

function CreateDoGRow(scaleSpaceRow){
    let dogRow = new cv.MatVector();
    for(let i = 0;i < gaussSequenceLength - 1; i++){
        let dog = CreateDoG(scaleSpaceRow.get(i), scaleSpaceRow.get(i+1));

        cv.normalize(dog, dog, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F);

        dogRow.push_back(dog);
    }
    return dogRow;
}

function CreateDoG(gauss1, gauss2){
    let dog = new cv.Mat();
    //let mask = new cv.Mat();
    //let dtype = -1;
    //cv.subtract(gauss1, gauss2, dog, mask, dtype);       
    cv.absdiff(gauss1, gauss2, dog);
    return dog;
}

function DrawKeypoints(image) {
    let orb = new cv.ORB(100);
    let kp = new cv.KeyPointVector();
    orb.detect(image, kp);
    cv.drawKeypoints(image,kp,image);
}
