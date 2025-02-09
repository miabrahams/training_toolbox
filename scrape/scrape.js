

import {promises as fsPromises} from 'fs';

// TODO: WIP code. Not sure what I used this for tbh.
/*
// FA
const fa_options = {
    cookies_file: "www.furaffinity.net.cookies.json",
    target_url: "https://www.furaffinity.net/gallery/honovy/",
    output: "claweddays.json"
}
const fa_selector = '#gallery-gallery > figure > p:nth-child(1) > a'
const fa_selectorFn = links => links.map(a => [a.href, a.innerText]);
const fa_nextLinkSelector = null;


// Grab titles as well as links!!
// Also: make visiting the links separate from finding them
async function fa_visitor(page, link, output, currentUrl = null) {
    try {
      await page.goto(link[0]);
      const text = await page.$eval('.submission-description', desc => desc.innerText)
      output[link[1]] = text;
    }
    catch (err) {
      console.log("Error: ", err)
    }
}


// Desuarchive
const ds_selector = 'a.post_file_filename';
const ds_selectorFn = (elements) => elements.map(el => {return {link: el.href, filename: el.innerText}});
const ds_nextLinkSelector = '::-p-text(Previous thread) a';
async function ds_visitor(page, link, textData, currentUrl = null){
  return null;
}

async function sd_labelVisitor(page, url) {
  page_id = url.split('/').slice(-1)[0];
}



// Aryion
async function ar_pageVisitor(page, url) {
  const data = await page.$$eval('div.post', (posts) =>
    posts.map((post) => {
      return {
        author: post.querySelector('.author strong').innerText,
        id: post.id,
        content: post.querySelector('.content').innerText,
      };
    })
  );

  const pageNumber = await page.$eval('.pagination strong', a => a.innerText);
  return {pageNumber, url, data};
}

const ar_nextLinkSelector = 'div.pagination > span > strong + .page-sep + a';

*/


/* Config */
import { promises as fsPromises } from 'fs';
import { chromium } from 'playwright';

// Configuration object
const config = {
    cookiesFile: null,  // Path to cookies file (optional)
    targetUrl: "https://aryion.com/forum/viewtopic.php?f=18&t=457",
    outputName: "aryion2.json",
    maxPages: 570,
    append: false,
    pageVisitor: async (page, url) => { // Refactor pageVisitor to be inside config
        const data = await page.$$eval('div.post', (posts) =>
            posts.map((post) => {
                return {
                    author: post.querySelector('.author strong').innerText,
                    id: post.id,
                    content: post.querySelector('.content').innerText,
                };
            })
        );

        const pageNumber = await page.$eval('.pagination strong', a => a.innerText);
        return { pageNumber, url, data };
    },
    nextLinkSelector: 'div.pagination > span > strong + .page-sep + a',
    itemVisitor: null, // Optional: Function to process each item
};

async function main() {
    const { targetUrl, maxPages, pageVisitor, nextLinkSelector, cookiesFile, outputName, append } = config;

    let cookies = null;
    if (cookiesFile) {
        try {
            cookies = JSON.parse(await fsPromises.readFile(`data/cookies/${cookiesFile}`, 'utf8'));
        } catch (err) {
            console.warn(`Could not read cookies file: ${cookiesFile}.  Continuing without cookies.`, err);
            cookies = null;
        }
    }

    console.log("Starting...");

    const browser = await chromium.launch();  // Launch Playwright Chromium
    const context = await browser.newContext(); // Create context, allows setting cookies
    const page = await context.newPage();        // Create a new page

    if (cookies) {
        await context.addCookies(cookies); // Playwright way of setting cookies
    }

    const outFile = `data/scrape/${outputName}`;
    let data = [];

    if (append) {
        console.log("Looking for existing data...");
        try {
            const fileContent = await fsPromises.readFile(outFile, 'utf8');
            data = JSON.parse(fileContent);
            console.log(`Loaded ${data.length} existing entries.`);
        } catch (err) {
            console.log('No existing data found. Creating new file.');
        }
    }

    // Disable image loading - Playwright method
    await page.route('**/*', route => {
        if (route.request().resourceType() === 'image')
            route.abort();
        else
            route.continue();
    });

    let currentUrl = targetUrl;

    for (let pageCount = 0; pageCount < maxPages; pageCount++) {
        if (!currentUrl) {
            console.log("Reached the end of the links.  Finishing.");
            break;
        }

        try {
            console.log(`Navigating to ${currentUrl}`);
            await page.goto(currentUrl);
            console.log(`Loaded page ${pageCount + 1}/${maxPages} - ${currentUrl}`);

            const pageData = await pageVisitor(page, currentUrl);
            data.push(pageData);

            try {
                currentUrl = await page.$eval(nextLinkSelector, el => el.href);
            } catch (err) {
                console.warn("Could not find next link.  Assuming end of sequence.", err);
                currentUrl = null;
            }

        } catch (err) {
            console.error(`Error processing ${currentUrl}: `, err);
            currentUrl = null;  // Stop if there's an error
        }
    }

    await browser.close();

    try {
        await fsPromises.writeFile(outFile, JSON.stringify(data, null, '  '));
        console.log(`Successfully wrote data to ${outFile}`);
    } catch (err) {
        console.error('The file could not be written.', err);
        console.error(JSON.stringify(data));  // Output data to console as fallback
    }
}

main().catch(console.error);