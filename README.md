# SmartTrafficMonitoring

## Member list
* 6310610941 - ชนาธินาถ พิศประเทือง - 6310610941@student.tu.ac.th
* 6310611014 - ชวันรัตน์ ทรัพย์เบญจภาคิน - 6310611014@student.tu.ac.th
* 6310611030 - ภัทรกร แก้วชูกุล - 6310611030@student.tu.ac.th
* 6310611089 - นิษฐา แก้วสมบัติ - 6310611089@student.tu.ac.th
* 6310682635 - ณัฐณิชา ฟักสังข์ - 6310682635@student.tu.ac.th
* 6310682726 - ณัฐนนท์ บุญเขตต์ - 6310682726@student.tu.ac.th

## About Project
SmartTrafficMonitoring is a project that apply AI for traffic monitoring.

### Built With
<!-- 
* [![Django][djangoproject.com]][Django-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
!-->
* [Django][Django-url]
* [Bootstrap][Bootstrap-url]
* [User Figma](https://www.figma.com/file/cAhjyeLYtVFkWdzGIe6PkM/CN332?node-id=0%3A1&t=YmoKge7GTPm9BFE9-1)

## Getting Started
### Prerequisites
You have to install software before using the project.

1. Download [Python](https://www.python.org/downloads/)
2. Install [Visual Studio Code](https://code.visualstudio.com/download)
3. Install [Redis](https://redis.io/docs/getting-started/installation/install-redis-on-mac-os/)

### Installation
1. Clone the repo
    ```sh
    git clone https://github.com/asnnat/SmartTrafficMonitoring.git
    ```
2. Change directory to the project
    ```sh
    cd SmartTrafficMonitoring/SmartTrafficMonitoring
    ```
3. Open the dirctory with Visual Studio Code
    ```sh
    code .
    ```

## Usage
1. Open git bash terminal in Visual Studio Code
2. Activate virtual environment
    ```sh
    source .venv/bin/activate
    ```
3. Install requirements for the project
    ```sh
    pip install -r requirements.txt
    ```
4. Run Django server in the 1st terminal
    ```sh
    python manage.py runserver
    ```
5. Run Redis server in the 2nd terminal
    ```sh
    redis-server
    ```
6. Run Celery worker in the 3rd terminal
    ```sh
    celery -A SmartTrafficMonitoring worker --pool=solo -l info
    ```
7.  Open link [click](http://127.0.0.1:8000/)

<!-- MARKDOWN LINKS & IMAGES -->
[djangoproject.com]: https://img.shields.io/badge/Djang0-35495E?style=for-the-badge&logo=django&logoColor=4FC08D
[Django-url]: https://www.djangoproject.com/
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
