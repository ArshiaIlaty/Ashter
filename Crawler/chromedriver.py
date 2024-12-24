#helper for downloading the chromedriver

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

# Use WebDriver Manager to download and manage ChromeDriver
driver = webdriver.Chrome(ChromeDriverManager().install())
