import { Component } from '@angular/core';
import { ActionSheetController } from '@ionic/angular';
import { Photo, PhotoService } from '../services/photo.service';
import {AlertController} from '@ionic/angular';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-tab2',
  templateUrl: 'tab2.page.html',
  styleUrls: ['tab2.page.scss']
})
export class Tab2Page {

  constructor(public photoService: PhotoService,
    public actionSheetController: ActionSheetController,
    public alertController: AlertController,
    private http: HttpClient) {}


    
    api_url="localhost:5000";

    ocrMaj(foto:string) {
      this.http.post('http://'+this.api_url+'/post_maj',{data: foto})
          .subscribe(
            (data) => {
              if (!data['txt_read'].replace(/[^0-9a-z]/gi, '')) {
                this.presentAlert("...nothing yet");
              }else{
              //this.presentAlert(data['txt_read'].replace('[^a]',"ZZZZZ").replace(/[^\na-z ]/gi, ''))
              console.log(data['txt_read']);
              this.presentAlert(data['txt_read'].replace(/[^\na-z ]/gi, '').replace(/\n/gi,"<br/>"));
              }
            },
      (error) =>{
        this.presentAlert("...I couldn't connect to the server\n[Error: "+error)
        }
        )};

      ocrMin(foto:string) {
        this.http.post('http://'+this.api_url+'/post_min',{data: foto})
            .subscribe(
              (data) => {
                if (!data['txt_read'].replace(/[^0-9a-z]/gi, '')) {
                  this.presentAlert("...nothing yet");
                }else{
                //this.presentAlert(data['txt_read'].replace('[^a]',"ZZZZZ").replace(/[^\na-z ]/gi, ''))
                console.log(data['txt_read']);
                this.presentAlert(data['txt_read'].replace(/[^\na-z ]/gi, '').replace(/\n/gi,"<br/>"));
                }
              },
        (error) =>{
          this.presentAlert("...I couldn't connect to the server\n[Error: "+error)
          }
          )};

    testget() {
      this.http.get('http://'+this.api_url+'/test_get')
          .subscribe((data) => {
          this.presentAlert(data['txt_read']);
        });
    }




  async ngOnInit() {
    await this.photoService.loadSaved();
  }

  async presentAlert(OCRresult: string) {

    const alert = await this.alertController.create({
      header: 'I read...',
      subHeader: '',
      message: OCRresult,
      buttons: ['OK'],
    });

    await alert.present();
  }



  async showPrompt() {
    let prompt = await this.alertController.create({
      header: 'Settings',
      message: "Choose API Url",
      inputs: [
        {
          name: 'url',
          placeholder: this.api_url,
        }
      ],
      buttons: [
        {
          text: 'Default',
          handler: data => {
            this.api_url = 'localhost:5000';
            console.log('Cancel clicked');
          }
        },{
          text: 'Save',
          handler: data => {
            console.log('Saved clicked');
            this.api_url = data.url;
          }
        }
      ]
    });
     await prompt.present();

  }

  public async showActionSheet(photo: Photo, position: number) {
    const actionSheet = await this.actionSheetController.create({

      buttons: [{
        text: 'OCR This! (uppercase)',
        role: 'destructive',
        icon: "scan-circle-outline",
        handler: () => {
          this.ocrMaj(photo.webviewPath);
        }
      },{
        text: 'OCR This! (lowercase)',
        role: 'destructive',
        icon: "scan-circle-outline",
        handler: () => {
          this.ocrMin(photo.webviewPath);
        }
      },{
        text: 'Delete',
        role: 'destructive',
        icon: 'trash',
        handler: () => {
          this.photoService.deletePicture(photo, position);
        }
      }, {
        text: 'Cancel',
        icon: 'close',
        role: 'cancel',
        handler: () => {
          // Nothing to do, action sheet is automatically closed
         }
      }]
    });
    await actionSheet.present();
  }
}
