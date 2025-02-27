import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NavgationComponent } from './navgation.component';

describe('NavgationComponent', () => {
  let component: NavgationComponent;
  let fixture: ComponentFixture<NavgationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [NavgationComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(NavgationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
