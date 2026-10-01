; Windows installer for the CorridorKey OBS filter.
;
; Installs the plugin into %ProgramData%\obs-studio\plugins, the path OBS for
; Windows actually scans (see install.ps1). It ships no models: they derive from
; checkpoints under a non-commercial licence, so the user generates them and
; puts them in the models folder this creates.
;
; Built in CI from the staged package directory:
;   iscc /DStage=dist\corridorkey-obs /DAppVersion=0.1.0 tools\package\installer.iss

#ifndef Stage
  #error Pass /DStage=<staged package dir>
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

[Setup]
AppId={{58B3BBF0-4406-4EC7-854B-A46DD515478D}
AppName=CorridorKey for OBS
AppVersion={#AppVersion}
AppPublisher=tangentstorm
AppPublisherURL=https://github.com/tangentstorm/corridorkey-obs
DefaultDirName={commonappdata}\obs-studio\plugins\corridorkey-obs
DisableDirPage=yes
DisableProgramGroupPage=yes
; ProgramData subfolders are not reliably writable by standard users once OBS
; has created them, and an installer asking for elevation is unsurprising.
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
InfoBeforeFile={#Stage}\NOTICE.txt
OutputDir=.
OutputBaseFilename=corridorkey-obs-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=CorridorKey for OBS

[Dirs]
Name: "{app}\data\models"

[Files]
Source: "{#Stage}\plugin\bin\64bit\corridorkey-obs.dll"; DestDir: "{app}\bin\64bit"; Flags: ignoreversion
Source: "{#Stage}\plugin\data\effects\corridorkey.effect"; DestDir: "{app}\data\effects"; Flags: ignoreversion
Source: "{#Stage}\plugin\data\models\NO-MODELS-YET.txt"; DestDir: "{app}\data\models"; Flags: ignoreversion
Source: "{#Stage}\NOTICE.txt"; DestDir: "{app}"; Flags: ignoreversion

[Code]
function ObsRunning: Boolean;
var
  Code: Integer;
begin
  Result := Exec(ExpandConstant('{cmd}'),
    '/C tasklist /FI "IMAGENAME eq obs64.exe" /NH | find /I "obs64.exe" >nul',
    '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
end;

// OBS holds the plugin DLL open, and the copy would fail with a file-in-use
// error that does not say why.
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  if ObsRunning then
    Result := 'OBS is running. Close it, then go back and try again.';
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpFinished then
    WizardForm.FinishedLabel.Caption := WizardForm.FinishedLabel.Caption + #13#10#13#10 +
      'No models are included. Until you put at least one in' + #13#10 +
      ExpandConstant('{app}\data\models') + #13#10 +
      'the filter passes video through untouched. See the README for how to make them.';
end;
