@ECHO OFF
cd /d C:\Users\CHANGLIHuang\Desktop\Kaggle

SETLOCAL enabledelayedexpansion
ECHO ================UPDATE FOLDER===============

SET COUNT=0
FOR /F %%a in ('dir /b') do (
 ECHO !COUNT!. %%a
 SET /A COUNT+=1
 
)
ENDLOCAL

SETLOCAL enabledelayedexpansion
SET /P NUM="対象ファルダの番号を入力してください：" 
SET COUNT2=0
FOR /F %%a in ('dir /b') do (
 IF !COUNT2!==!NUM! (
  git add %%a
  git commit -m "Learning"
  git push origin master
 )
 SET /A COUNT2+=1
)

PAUSE