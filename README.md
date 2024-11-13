<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="/Image_Readme.png" alt="Logo" width="500" height="300">
  </a>

  <h2 align="center">Shopping baskets tracking with AprilTags</h2>




</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#packages">Packages</a></li>
    <li><a href="#unit_testing">Unit_testing</a></li>
    <li><a href="#links">Links</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project was part of an internship, the goal is to track shopping baskets in real time using AprilTags placed on them.
The initial constraint was not to train a model for detecting shopping carts.

The chosen method is to detect AprilTags with the "pupil-apriltag" library, to segment the image and check if the tag coordinates overlap with one of the segmented masks, if yes we follow the mask.

This method solves the problem related to difficult detection due to the small size of the tags, the light and the speed of movement, in addition to the times when the tag disappears due to the customer's position. 
Tracking the mask is more reliable over the entire duration of the customer's presence. This also allows us to assign a unique ID to each basket.



<!-- Prerequisites -->
## Prerequisites

Install the differents librairies from the requirements file

  ```sh
  pip install -r requirements.txt
  ```


<!-- Packages -->
## Packages

Packages & librairies used
* [OpenCV](https://opencv.org/)
* [Pupil_AprilTag](https://pypi.org/project/pupil-apriltags/)
* [Ultralytics](https://www.ultralytics.com/)
* [Numpy](https://numpy.org/)
* [Requests](https://pypi.org/project/requests/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- unit_testing -->
## Unit_testing

Test framework
* [Unittest](https://docs.python.org/3/library/unittest.html)


<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- LINKS -->
## Links

Github : [https://github.com/ValerioCann/](https://github.com/your_username/repo_name)

Project Link: [https://github.com/ValerioCann/Baskets_Tracking_AprilTags](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
