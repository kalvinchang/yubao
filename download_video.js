
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function getElementSrcByXPath(xpath) {
    let element = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
    if (element) {
        console.log(element.src);
        return element.src;
    } else {
        console.log("Element not found.");
        return null;
    }
}

function clickButtonByXPath(xpath) {
    let button = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
    if (button) {
        button.click();
        console.log("Button clicked.");
    } else {
        console.log("Button not found.");
    }
}

// get URL of each video in a series and move onto next page
async function scrapeVideo(videoPath, buttonPath, maxNum) {
    const vids = [];
    vids.push(getElementSrcByXPath(videoPath));

    for (let i = 0; i < maxNum - 1; i++) {
        clickButtonByXPath(buttonPath);
        await delay(15000); // Wait for 10 seconds
        vids.push(getElementSrcByXPath(videoPath));
    }

    return vids;
}


async function downloadVideo(city, vidsArray) {
    // TODO: easy way to detect rate limiting - the URL is the same as the previous one
    for (let i = 0; i < vidsArray.length; i++) {
        const vidUrl = vidsArray[i];
        const response = await fetch(vidUrl);
        const blob = await response.blob();
        const downloadLink = document.createElement('a');
        const url = URL.createObjectURL(blob);
        downloadLink.href = url;
        downloadLink.download = `${city}_${String(i + 1).padStart(4, '0')}.mp4`;
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        URL.revokeObjectURL(url);
        await delay(10000); // 10 second delay
    }
}

// TODO: use this when you want words
async function clickWordTab(){
    const word = document.querySelector("div[data-node-key='sentence']");
    word.click();
    await delay(500);
}


async function clickSentenceTab(){
    const sentence = document.querySelector("div[data-node-key='sentence']");
    sentence.click();
    await delay(500);
}

async function clickVideoIcon(videoIconPath) {
    clickButtonByXPath(videoIconPath);
    await delay(20000);
}


const cityXPath = '//*[@id="root"]/section/div[1]/div/div[1]/div[1]/div[1]/div[1]/div[1]/div';
const videoPath = '//*[@id="videoContentShort"]/div/div/div[1]/video';
const nextButtonXPath = '//*[contains(@class, "ant-modal-body")]/div/div[2]/div[2]/button[2]';
// the current tab will have tabindex, "0"
// if you want the character tab, you need to click the character tab first (same with sentences)
const videoIconPath = '//*[contains(@tabindex, "0")]//*[contains(@class, "ant-table-row")][1]/td[5]/span';
const numSentences = 50;

// TODO: handle cases where no word is there

const numWords = 1200;
// get the city
const city = document.evaluate(cityXPath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue.innerText;

// first, click on the sentence tab
clickSentenceTab()
    // then click on the video path
    .then(() => clickVideoIcon(videoIconPath))
    // then get ALL of the video links by using the paginator
    .then(() => scrapeVideo(videoPath, nextButtonXPath, numSentences))
    // temporarily save the video links
    .then(links => {
        // Convert the links array into a single string with each link on a new line
        const data = links.join('\n') + '\n';

        // Create a Blob object with the data
        const blob = new Blob([data], { type: 'text/plain' });

        // Create a temporary anchor element
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob); // Create a URL for the Blob
        a.download = `${city}_video_links.txt`; // Set the desired file name

        // Programmatically click the anchor to trigger the download
        a.click();

        // Clean up by revoking the object URL
        URL.revokeObjectURL(a.href);

        return links;
    })
    // finally, download the video using the links
    .then(links => downloadVideo(city, links));
