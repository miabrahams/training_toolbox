const puppeteer = require('puppeteer');

(async () => {
	// Initiate the browser
	const browser = await puppeteer.launch();

	// Create a new page with the default browser context
	const page = await browser.newPage();

	// Go to the target website
	await page.goto('https://google.com');

	// Get pages HTML content
	const content = await page.content();
	console.log(content);

	// Closes the browser and all of its pages
	await browser.close();
})();