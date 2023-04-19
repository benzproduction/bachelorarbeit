import Link from 'next/link';

type LegalItemProps = {
  title: string;
  link: string;
  [x: string]: any;
};

const LegalItem: React.FC<LegalItemProps> = ({ title, link, ...props }) => {
  return (
    <span className="p-1 text-sm text-white">
      <Link href={link}>
        <span className="no-underline hover:text-data-yellow-value" {...props}>
          {title}
        </span>
      </Link>
    </span>
  );
};

type Props = {
  impressum?: boolean;
};

const Footer: React.FC<Props> = ({ impressum = true }) => {
  const year = new Date().getFullYear();
  return (
    <div
      id="footer"
      className="flex flex-wrap items-center justify-between bg-colors-background-default-dark px-1"
    >
      <div className="flex">
        <span className="p-1 text-sm text-white">
          &#xA9; {year} PwC. All rights reserved. PwC refers to the PwC network
          and/or one or more of its member firms, each of which is a separate
          legal entity.
        </span>
      </div>

      {impressum && (
        <div className="flex">
          <LegalItem
            title="Disclaimer"
            link="https://www.pwc.de/de/disclaimer.html"
          />
          <LegalItem
            title="Imprint"
            link="https://www.pwc.de/de/impressum.html"
          />
          <LegalItem
            title="Terms of use"
            link="https://www.pwc.de/de/nutzungsbedingungen.html"
          />
          <LegalItem
            title="Privacy policy"
            link="https://pwc.de/de/datenschutzerklaerung-fuer-mandanten/datenschutzhinweise-pricewaterhousecoopers-gmbh-wirtschaftspruefungsgesellschaft.html"
          />
        </div>
      )}
    </div>
  );
};

export default Footer;
